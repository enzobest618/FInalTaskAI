import os
import json
import psycopg2
import numpy as np
from scipy.optimize import linprog
from dotenv import load_dotenv
from poke_env.data import GenData

load_dotenv()

def apply_variance_penalty(payoff_matrix: np.ndarray, move_accuracies: list, penalty_factor: float = 0.08) -> np.ndarray:
    """
    3. Variance penalty: reduces the payoffs (Win Rate) of Pokémon 
    that rely on moves with less than 90% accuracy.
    Models risk aversion in 1v1 where a miss is fatal.
    """
    adjusted_matrix = np.copy(payoff_matrix)
    num_strats = len(move_accuracies)
    
    for i in range(num_strats):
        if move_accuracies[i] < 0.90:
            # If P1 plays an inaccurate Pokémon, its expected Win Rate decreases
            adjusted_matrix[i, :] -= penalty_factor
            # If P2 plays an inaccurate Pokémon, its Win Rate decreases (so P1's payoff increases, zero-sum game)
            adjusted_matrix[:, i] += penalty_factor
            
    return np.clip(adjusted_matrix, 0.0, 1.0)

def solve_nash_equilibrium(payoff_matrix: np.ndarray):
    """
    2. Resolution of the Nash Equilibrium in mixed strategies (Minimax).
    
    MATHEMATICAL EXPLANATION OF LINEAR PROGRAMMING:
    We seek to Maximize our guaranteed minimum payoff 'v'.
    In standard linear programming (which only knows how to Minimize), we Minimize '-v'.
    
    The decision vector 'x' will be organized as follows: [v, p1, p2, p3, p4, p5]
    Where 'v' is the value of the game, and 'p_i' the probability of playing team i.
    """
    num_strats = payoff_matrix.shape[0] # Number of strategies (teams)

    # --- OBJECTIVE FUNCTION (c) ---
    # We want to minimize -v (so c[0] = -1) and we ignore the probabilities in the objective (c[1:] = 0).
    c = np.zeros(num_strats + 1)
    c[0] = -1.0 

    # --- INEQUALITY CONSTRAINTS (A_ub * x <= b_ub) ---
    # For each strategy 'j' that the opponent can play, our expected payoff must be at least 'v'.
    # Formula: Sum(p_i * Payoff_ij) >= v  ==>  v - Sum(p_i * Payoff_ij) <= 0
    # So, for 'v', the coeff is 1. For 'p_i', the coeff is -Payoff_ij.
    A_ub = np.zeros((num_strats, num_strats + 1))
    A_ub[:, 0] = 1.0 # Column 0 corresponds to 'v'
    A_ub[:, 1:] = -payoff_matrix.T # We transpose so that the opponent's columns become rows of constraints
    
    b_ub = np.zeros(num_strats) # Everything must be <= 0

    # --- EQUALITY CONSTRAINTS (A_eq * x == b_eq) ---
    # The sum of our probabilities (p1 + p2 + ... + p5) must be equal to 100% (1.0)
    A_eq = np.zeros((1, num_strats + 1))
    A_eq[0, 1:] = 1.0 # Coeff 1 for all probs, 0 for 'v'
    b_eq = np.array([1.0])

    # --- BOUNDS ---
    # 'v' can be any real value: (None, None)
    # Each probability 'p_i' must be between 0% and 100%: (0.0, 1.0)
    bounds = [(None, None)] + [(0.0, 1.0)] * num_strats

    # Resolution with SciPy's modern 'highs' method (more stable than the old simplex algorithm)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        game_value = res.x[0]      # v (Our guaranteed expected Win Rate)
        probabilities = res.x[1:]  # [p1, p2, p3, p4, p5]
        return game_value, probabilities
    else:
        raise ValueError("Impossible to find an equilibrium: " + res.message)

def save_nash_to_db(game_value, optimal_probs, pokemon_names):
    """Saves the Nash equilibrium results to NeonDB."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("⚠️ Error: DATABASE_URL not defined in .env.")
        return
        
    print("🔄 Attempting to connect to NeonDB for export...")
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nash_results (
                id SERIAL PRIMARY KEY,
                game_value FLOAT,
                optimal_probabilities JSON,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        probs_dict = {name: float(prob) for name, prob in zip(pokemon_names, optimal_probs) if prob > 0.001}
        
        cursor.execute("""
            INSERT INTO nash_results (game_value, optimal_probabilities)
            VALUES (%s, %s::json)
        """, (float(game_value), json.dumps(probs_dict)))
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Nash Equilibrium results saved to NeonDB ('nash_results' table)!")
    except Exception as e:
        print(f"❌ DB Error: {e}")

def get_real_data_from_db():
    """Retrieves raw results from NeonDB and builds the payoff and accuracy matrix."""
    print("📥 Retrieving played matches from NeonDB (tournament_results)...")
    db_url = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    # 1. Retrieving all aggregated battles
    cursor.execute("""
        SELECT p1_name, p2_name, winner, COUNT(*) 
        FROM tournament_results 
        GROUP BY p1_name, p2_name, winner
    """)
    battles = cursor.fetchall()
    
    # 2. Retrieving the moves of each Pokémon for accuracy
    cursor.execute('SELECT "name", "moves" FROM "pokemons"')
    pokemon_moves = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    if not battles:
        print("❌ No battles found in tournament_results. Let bot.py run for a bit!")
        return None
        
    # Identify all Pokémon that have played
    played_pokemons = set()
    for p1, p2, _, _ in battles:
        played_pokemons.add(p1)
        played_pokemons.add(p2)
        
    pokemon_names = sorted(list(played_pokemons))
    name_to_idx = {name: i for i, name in enumerate(pokemon_names)}
    num_strats = len(pokemon_names)
    
    # Initialize matrices
    wins = np.zeros((num_strats, num_strats))
    matches = np.zeros((num_strats, num_strats))
    
    for p1, p2, winner, count in battles:
        i, j = name_to_idx[p1], name_to_idx[p2]
        matches[i, j] += count
        matches[j, i] += count
        
        if winner == 'p1':
            wins[i, j] += count
        elif winner == 'p2':
            wins[j, i] += count
        else: # Draw
            wins[i, j] += count * 0.5
            wins[j, i] += count * 0.5
            
    # Calculation of the Payoff Matrix (Expected Win Rate)
    # If 2 Pokémon have never fought, we put 0.5 (50/50 by default)
    payoff_matrix = np.full((num_strats, num_strats), 0.5)
    for i in range(num_strats):
        for j in range(num_strats):
            if matches[i, j] > 0:
                payoff_matrix[i, j] = wins[i, j] / matches[i, j]
                
    # 3. Calculation of average accuracies
    gen1_data = GenData.from_gen(1)
    moves_dict = {name: moves for name, moves in pokemon_moves}
    
    move_accuracies = []
    for name in pokemon_names:
        moves = moves_dict.get(name, [])
        if isinstance(moves, str): moves = json.loads(moves)
            
        total_acc, count_moves = 0, 0
        for move_name in moves:
            move_info = gen1_data.moves.get(move_name.replace('-', '').lower(), {})
            acc = move_info.get("accuracy", 100)
            if acc is True or not isinstance(acc, (int, float)): acc = 100.0
            total_acc += acc
            count_moves += 1
            
        avg_acc = (total_acc / count_moves / 100.0) if count_moves > 0 else 1.0
        move_accuracies.append(avg_acc)
        
    return payoff_matrix, move_accuracies, pokemon_names

def main():
    print("🚀 Launching Nash solver (Connected to REAL results from bot.py) ...")
    
    data = get_real_data_from_db()
    if not data:
        return
        
    payoff_matrix, move_accuracies, pokemon_names = data
    
    print(f"📊 {len(pokemon_names)}x{len(pokemon_names)} payoff matrix generated successfully!")
    
    adjusted_matrix = apply_variance_penalty(payoff_matrix, move_accuracies, penalty_factor=0.08)

    print("\n🧠 4. FINAL OUTPUT: Nash Equilibrium Resolution...")
    try:
        # 2. Using linprog to find the equilibrium
        val, optimal_probs = solve_nash_equilibrium(adjusted_matrix)
        
        print(f"\n=> Guaranteed Minimum Win Rate (Game Value): {val*100:.2f}%")
        print("\n=> OPTIMAL STRATEGY DISTRIBUTION (META):\nIn a series of duels, play:")
        for i, prob in enumerate(optimal_probs):
            # We hide probabilities infinitely close to zero due to floats
            if prob > 0.001: 
                print(f"   - {pokemon_names[i]} at {prob * 100:.1f}%")
                
        # Export to NeonDB
        save_nash_to_db(val, optimal_probs, pokemon_names)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()