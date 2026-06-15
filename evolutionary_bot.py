import os
import json
import random
import psycopg2
from dotenv import load_dotenv

from poke_env.data import GenData
from fitness_engine import FitnessEngine

load_dotenv()

# --- GENETIC ALGORITHM CONFIGURATION ---
NUM_MODELS = 20000        # Number of times the entire evolution is restarted (20000 models)
MAX_GENERATIONS = 10000   # No strict limit (will be cut by the evolution plateau)
MIN_GENERATIONS = 20      # Minimum generations before allowing stop
PLATEAU_PATIENCE = 5      # Stops the model if the best remains identical for 5 generations
MATCHES_PER_IND = 3       # Number of matches played per individual to evaluate its "Fitness"
MUTATION_RATE = 0.2       # 20% chance to mutate

USELESS_MOVES = {
    "splash", "teleport", "roar", "whirlwind", "tail-whip", "growl", 
    "leer", "string-shot", "sand-attack", "smokescreen", "kinesis", "flash",
    "focus-energy", "bide", "poison-gas"
}

def get_valid_pokemons():
    """Retrieves all Pokémon and their moves from NeonDB to build the gene pool."""
    db_url = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    cursor.execute('SELECT "name", "moves", "stats" FROM "pokemons"')
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    pokemon_pool = {}
    for name, moves, stats in results:
        if isinstance(moves, str):
            moves = json.loads(moves)
        if isinstance(stats, str):
            stats = json.loads(stats)
        filtered_moves = [m for m in moves if m not in USELESS_MOVES]
        if not filtered_moves:
            filtered_moves = moves
        bst = sum(stats.values()) if stats else 0
        pokemon_pool[name] = {
            "moves": filtered_moves,
            "stats": stats,
            "bst": bst
        }
    return pokemon_pool

def generate_random_individual(pokemon_pool):
    """Generates a random individual (a Pokémon + 4 moves)."""
    species = random.choice(list(pokemon_pool.keys()))
    valid_moves = pokemon_pool[species]["moves"]
    num_moves = min(4, len(valid_moves))
    moves = random.sample(valid_moves, num_moves)
    
    return {
        "species_id": species,
        "bst": pokemon_pool[species]["bst"],
        "base_stats": pokemon_pool[species]["stats"],
        "moves": moves,
        "raw_wins": 0.0,
        "fitness": 0
    }

def save_generation_to_db(run_id, gen_id, best_ind):
    """Saves the best individual of a generation in NeonDB."""
    db_url = os.environ.get("DATABASE_URL")
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolutionary_results (
                id SERIAL PRIMARY KEY,
                run_id INTEGER,
                generation INTEGER,
                best_fitness INTEGER,
                best_species VARCHAR(50),
                best_moves JSON,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Automatically updates the table if it's from the old version
        cursor.execute("ALTER TABLE evolutionary_results ADD COLUMN IF NOT EXISTS run_id INTEGER;")
        cursor.execute("ALTER TABLE evolutionary_results ALTER COLUMN best_fitness TYPE FLOAT;")
        
        cursor.execute("""
            INSERT INTO evolutionary_results (run_id, generation, best_fitness, best_species, best_moves)
            VALUES (%s, %s, %s, %s, %s::json)
        """, (run_id, gen_id, best_ind["fitness"], best_ind["species_id"], json.dumps(best_ind["moves"])))
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ DB Error (Generation {gen_id}): {e}")

def main():
    
    pokemon_pool = get_valid_pokemons()
    POPULATION_SIZE = len(pokemon_pool) # Adapts to the exact number of Pokémon (151)
    
    # Loading Gen 1 data to retrieve accuracies
    gen1_data = GenData.from_gen(1)
    moves_db = {}
    for move_id, move_info in gen1_data.moves.items():
        moves_db[move_id] = {
            "power": move_info.get("basePower", 0),
            "accuracy": move_info.get("accuracy", 100),
            "category": move_info.get("category", "Physical")
        }
    
    # Initialization of the ultra-fast heuristic engine
    engine = FitnessEngine(moves_db=moves_db)

    print(f"🧬 Starting Evolutionary Algorithm! ({NUM_MODELS} models planned, {POPULATION_SIZE} individuals per generation)")

    # MODELS LOOP
    for run_id in range(1, NUM_MODELS + 1):
        print(f"\n🚀 === STARTING EVOLUTIONARY MODEL {run_id}/{NUM_MODELS} ===")
        
        # 1. POPULATION INITIALIZATION (Turn 1: Exactly the 151 Pokémon)
        population = []
        for species, valid_moves in pokemon_pool.items():
            num_moves = min(4, len(valid_moves["moves"]))
            moves = random.sample(valid_moves["moves"], num_moves)
            population.append({
                "species_id": species,
                "bst": valid_moves["bst"],
                "base_stats": valid_moves["stats"],
                "moves": moves,
                "raw_wins": 0.0,
                "fitness": 0
            })
            
        best_traits_history = []

        for gen in range(1, MAX_GENERATIONS + 1):
            # Reset for the new generation
            for ind in population:
                ind["raw_wins"] = 0.0
                ind["fitness"] = 0.0

            # 2. FITNESS EVALUATION (Via the new fast engine)
            engine.evaluate_population(population, matches_per_ind=MATCHES_PER_IND)
                
            # 3. SELECTION
            population.sort(key=lambda x: x["fitness"], reverse=True)
            best_ind = population[0]
            print(f"🏆 Gen {gen} : {best_ind['species_id']} (BST: {best_ind['bst']}) avec {best_ind['moves']} | Fitness: {best_ind['fitness']:.2f}")
            
            # --- EVOLUTION PLATEAU CHECK ---
            # We save the species and moves of the current best
            current_best_traits = (best_ind["species_id"], tuple(sorted(best_ind["moves"])))
            best_traits_history.append(current_best_traits)
            
            if gen >= MIN_GENERATIONS and len(best_traits_history) >= PLATEAU_PATIENCE:
                # Checks if the last 5 best have exactly the same combination (Species + Moves)
                recent_traits = best_traits_history[-PLATEAU_PATIENCE:]
                if len(set(recent_traits)) == 1:
                    print(f"🛑 Evolution plateau reached! Champion hasn't changed for {PLATEAU_PATIENCE} generations. Ending model {run_id}.")
                    # We save ONLY the final champion in the database
                    save_generation_to_db(run_id, gen, best_ind)
                    break # Stop this model's evolution and move to next run_id
            
            # If we reach the very last generation without hitting a plateau, save anyway
            if gen == MAX_GENERATIONS:
                save_generation_to_db(run_id, gen, best_ind)
            
            # 4. CROSSOVER AND MUTATION
            parents = population[:POPULATION_SIZE // 2]
            next_generation = []
            
            for p in parents:
                next_generation.append({
                    "species_id": p["species_id"],
                    "bst": p["bst"],
                    "base_stats": p["base_stats"],
                    "moves": list(p["moves"]),
                    "raw_wins": 0.0,
                    "fitness": 0
                })
                
            while len(next_generation) < POPULATION_SIZE:
                p1, p2 = random.sample(parents, 2)
                child_species = random.choice([p1["species_id"], p2["species_id"]])
                
                base_parent = p1 if child_species == p1["species_id"] else p2
                child_moves = list(base_parent["moves"])
                
                if random.random() < MUTATION_RATE:
                    mutation_type = random.choice(["species", "moves"])
                    
                    if mutation_type == "species":
                        child_species = random.choice(list(pokemon_pool.keys()))
                        valid_moves = pokemon_pool[child_species]["moves"]
                        child_moves = random.sample(valid_moves, min(4, len(valid_moves)))
                    elif mutation_type == "moves":
                        valid_moves = pokemon_pool[child_species]["moves"]
                        if len(valid_moves) > len(child_moves):
                            available_new_moves = [m for m in valid_moves if m not in child_moves]
                            if available_new_moves:
                                new_move = random.choice(available_new_moves)
                                idx_to_replace = random.randint(0, len(child_moves) - 1)
                                child_moves[idx_to_replace] = new_move
                                
                next_generation.append({
                    "species_id": child_species,
                    "bst": pokemon_pool[child_species]["bst"],
                    "base_stats": pokemon_pool[child_species]["stats"],
                    "moves": child_moves,
                    "raw_wins": 0.0,
                    "fitness": 0
                })
                
            population = next_generation

    print(f"🏁 The {NUM_MODELS} Models of the Evolutionary Algorithm are finished!")

if __name__ == "__main__":
    main()