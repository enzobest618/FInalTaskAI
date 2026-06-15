import json
import urllib.request
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_pokemon_data(limit=151):
    url = f"https://pokeapi.co/api/v2/pokemon?limit={limit}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.load(response)
        
    pokemons = []
    print(f"⏳ Récupération des données et attaques de la génération 1 pour {limit} Pokémon (cela peut prendre un peu de temps)...")
    for result in data["results"]:
        pokemon_url = result["url"]
        req_poke = urllib.request.Request(pokemon_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_poke) as response_poke:
            poke_data = json.load(response_poke)
            
        gen1_moves = []
        for move_entry in poke_data["moves"]:
            # Vérifie si l'attaque était disponible dans les versions Rouge/Bleu ou Jaune
            is_gen1 = any(
                vg["version_group"]["name"] in ["red-blue", "yellow"]
                for vg in move_entry["version_group_details"]
            )
            if is_gen1:
                gen1_moves.append(move_entry["move"]["name"])
                
        base_stats = {}
        for stat_info in poke_data["stats"]:
            stat_name = stat_info["stat"]["name"]
            val = stat_info["base_stat"]
            if stat_name == "hp": base_stats["hp"] = val
            elif stat_name == "attack": base_stats["atk"] = val
            elif stat_name == "defense": base_stats["def"] = val
            elif stat_name == "special-attack": base_stats["spa"] = val
            elif stat_name == "special-defense": base_stats["spd"] = val
            elif stat_name == "speed": base_stats["spe"] = val
            
        pokemons.append({
            "name": poke_data["name"],
            "moves": gen1_moves,
            "stats": base_stats
        })
    return pokemons

def save_to_neondb(pokemons_data):
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("⚠️ Erreur : La variable d'environnement DATABASE_URL n'est pas définie.")
        return

    try:
        # Connexion à NeonDB
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Création de la table si elle n'existe pas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "pokemons" (
                "id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                "name" text UNIQUE,
                "moves" json,
                "stats" json
            )
        """)
        
        cursor.execute('ALTER TABLE "pokemons" ADD COLUMN IF NOT EXISTS "stats" json')
        
        # Mise à jour ou Insertion des données (Upsert manuel pour éviter les problèmes de contrainte)
        for pk_data in pokemons_data:
            moves_json = json.dumps(pk_data["moves"])
            stats_json = json.dumps(pk_data["stats"])
            name = pk_data["name"]
            
            # Met à jour le Pokémon s'il est déjà en base
            cursor.execute('UPDATE "pokemons" SET "moves" = %s::json, "stats" = %s::json WHERE "name" = %s', (moves_json, stats_json, name))
            
            # Si aucune ligne n'a été modifiée, on l'insère
            if cursor.rowcount == 0:
                cursor.execute('INSERT INTO "pokemons" ("name", "moves", "stats") VALUES (%s, %s::json, %s::json)', (name, moves_json, stats_json))
            
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Succès : Les {len(pokemons_data)} Pokémon et leurs attaques ont été enregistrés dans NeonDB.")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde dans la base de données : {e}")

def main():
    pokemons_data = get_pokemon_data()
    for pk in pokemons_data:
        print(f"{pk['name']} : {len(pk['moves'])} attaques (Gen 1)")
    save_to_neondb(pokemons_data)

if __name__ == "__main__":
    main()