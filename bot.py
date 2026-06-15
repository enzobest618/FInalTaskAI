import os
import json
import random
import asyncio
import psycopg2
from dotenv import load_dotenv
import logging

from poke_env.player import Player
from poke_env import ServerConfiguration
from poke_env.teambuilder import Teambuilder

load_dotenv()

# Configuration to point to the "showdown" container in the Docker network
DockerServerConfiguration = ServerConfiguration(
    "ws://showdown:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?"
)

# Dynamic Teambuilder to change team for each battle without recreating the player
class DynamicTeambuilder(Teambuilder):
    def __init__(self):
        self.current_team = ""
        
    def yield_team(self):
        return self.current_team

# Bot maximizing damage (base power)
class MaxDamageBot(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            # Acts like a real player by calculating expected damage (Power * Accuracy)
            def expected_damage(m):
                power = m.base_power
                acc = 100 if m.accuracy is True else (m.accuracy or 100)
                return power * (acc / 100.0)
                
            best_move = max(battle.available_moves, key=expected_damage, default=battle.available_moves[0])
            return self.create_order(best_move)
        return self.choose_random_move(battle)

# --- NEW: Exclusion lists to reduce combinatorial explosion ---
USELESS_MOVES = {
    "splash", "teleport", "roar", "whirlwind", "tail-whip", "growl", 
    "leer", "string-shot", "sand-attack", "smokescreen", "kinesis", "flash",
    "focus-energy", "bide", "poison-gas"
}

def get_pokemon_pool():
    print("📥 Retrieving Pokémon from NeonDB...")
    db_url = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    cursor.execute('SELECT "name", "moves" FROM "pokemons"')
    results = cursor.fetchall()
    
    pokemon_pool = {}
    for name, moves in results:
        if isinstance(moves, str):
            moves = json.loads(moves)
            
        # Removing useless moves
        filtered_moves = [m for m in moves if m not in USELESS_MOVES]
        
        # Make sure moves remain, otherwise keep original
        if len(filtered_moves) == 0:
            filtered_moves = moves
            
        pokemon_pool[name] = filtered_moves
            
    cursor.close()
    conn.close()
    print(f"✅ {len(pokemon_pool)} Pokémon retrieved.")
    return pokemon_pool

def save_results_to_db(results_data):
    if not results_data:
        return
        
    print("💾 Massively saving results to NeonDB...", flush=True)
    try:
        db_url = os.environ.get("DATABASE_URL")
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tournament_results (
                id SERIAL PRIMARY KEY,
                p1_name VARCHAR(100),
                p1_moves JSON,
                p2_name VARCHAR(100),
                p2_moves JSON,
                winner VARCHAR(100)
            )
        """)
        
        insert_query = """
            INSERT INTO tournament_results (p1_name, p1_moves, p2_name, p2_moves, winner)
            VALUES (%s, %s::json, %s, %s::json, %s)
        """
        
        # Preparing data for batch insertion (highly optimized)
        data_to_insert = [
            (
                r["p1_name"], json.dumps(r["p1_moves"]),
                r["p2_name"], json.dumps(r["p2_moves"]),
                r["winner"]
            )
            for r in results_data
        ]
        
        cursor.executemany(insert_query, data_to_insert)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ {len(results_data)} results successfully saved to NeonDB!", flush=True)
    except Exception as e:
        print(f"❌ Error saving to NeonDB: {e}", flush=True)

async def wait_for_showdown():
    print("⏳ Waiting for Showdown server to start...")
    while True:
        try:
            reader, writer = await asyncio.open_connection("showdown", 8000)
            writer.close()
            await writer.wait_closed()
            print("✅ Showdown server ready!")
            break
        except OSError:
            await asyncio.sleep(2)

async def main():
    await wait_for_showdown()
    
    pokemon_pool = get_pokemon_pool()
    pokemon_names = list(pokemon_pool.keys())
    
    if len(pokemon_names) < 2:
        print("❌ Not enough Pokémon to start a tournament.")
        return

    tb1 = DynamicTeambuilder()
    tb2 = DynamicTeambuilder()
    
    player1 = MaxDamageBot(battle_format="gen1customgame", team=tb1, server_configuration=DockerServerConfiguration)
    player2 = MaxDamageBot(battle_format="gen1customgame", team=tb2, server_configuration=DockerServerConfiguration)

    # Hide harmless WARNINGs directly on bot loggers
    player1.logger.setLevel(logging.ERROR)
    player2.logger.setLevel(logging.ERROR)
    logging.getLogger("poke-env").setLevel(logging.ERROR)

    results_data = []
    total_battles = 100000
    batch_size = 5  # Very frequent saving to see real-time data
    print(f"🏆 Launching Monte-Carlo tournament! Saving every {batch_size} battles.")
    
    for match_id in range(1, total_battles + 1):
        # 1. Random selection of 2 distinct Pokémon among the 151
        name1, name2 = random.sample(pokemon_names, 2)
        
        # 2. Random selection of their moves (max 4)
        moves1 = pokemon_pool[name1]
        combo1 = random.sample(moves1, min(4, len(moves1)))
        
        moves2 = pokemon_pool[name2]
        combo2 = random.sample(moves2, min(4, len(moves2)))
        
        # 3. Team building (packed format)
        def build_team(name, combo):
            species_id = name.replace('-', '')
            moves_id = ','.join([m.replace('-', '') for m in combo])
            return f"|{species_id}|||{moves_id}|||||||"
            
        it1 = {"name": name1, "combo": combo1, "team": build_team(name1, combo1)}
        it2 = {"name": name2, "combo": combo2, "team": build_team(name2, combo2)}
        
        # Dynamic team modification
        tb1.current_team = it1["team"]
        tb2.current_team = it2["team"]
        
        # More frequent progress display
        if match_id % 10 == 0 or match_id == 1 or match_id == total_battles:
            print(f"⚔️ Battle {match_id}/{total_battles} : {it1['name']} VS {it2['name']}", flush=True)
        
        player1.reset_battles()
        player2.reset_battles()
        
        # gen1customgame does not limit any move and allows 1v1 format 
        await player1.battle_against(player2, n_battles=1)
        
        if player1.n_won_battles > 0:
            winner = "p1"
        elif player2.n_won_battles > 0:
            winner = "p2"
        else:
            winner = "draw" # Draw or turn limit exceeded
        
        results_data.append({
            "p1_name": it1["name"],
            "p1_moves": it1["combo"],
            "p2_name": it2["name"],
            "p2_moves": it2["combo"],
            "winner": winner
        })

        # Batch saving to avoid losing everything in case of interruption
        if len(results_data) >= batch_size:
            save_results_to_db(results_data)
            results_data.clear()

    print("🏁 Tournament finished! Saving results...")
    if results_data:
        save_results_to_db(results_data)

if __name__ == "__main__":
    asyncio.run(main()) 