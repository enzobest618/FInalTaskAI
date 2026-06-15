import os
import json
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

load_dotenv()

def save_test_results_to_db(results_df):
    db_url = os.environ.get("DATABASE_URL")
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_test_results_1v1 (
                id SERIAL PRIMARY KEY,
                p1_name VARCHAR(100),
                p2_name VARCHAR(100),
                actual_winner VARCHAR(10),
                predicted_winner VARCHAR(10),
                is_correct BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        insert_query = """
            INSERT INTO ai_test_results_1v1 (p1_name, p2_name, actual_winner, predicted_winner, is_correct)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        data_to_insert = [
            (
                row['p1_name'], 
                row['p2_name'], 
                'p1' if row['target'] == 1 else 'p2',
                'p1' if row['prediction'] == 1 else 'p2',
                bool(row['is_correct'])
            )
            for _, row in results_df.iterrows()
        ]
        
        cursor.executemany(insert_query, data_to_insert)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ {len(data_to_insert)} verification results (True/False) saved in ai_test_results_1v1!")
    except Exception as e:
        print(f"❌ DB Error (Test Results 1v1): {e}")

def fetch_data():
    db_url = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    # 1. Global Win Rates from bot.py
    print("📥 1/5 - Retrieving global Win Rates (bot.py)...")
    cursor.execute("""
        SELECT p1_name, 
               SUM(CASE WHEN winner = 'p1' THEN 1 ELSE 0 END) as wins,
               COUNT(*) as total
        FROM tournament_results
        GROUP BY p1_name
    """)
    p1_stats = cursor.fetchall()
    winrates = {name: (wins / total if total > 0 else 0.5) for name, wins, total in p1_stats}

    # 2. Meta and Probabilities from nash_solver.py
    print("📥 2/5 - Retrieving Nash Equilibrium...")
    cursor.execute("SELECT optimal_probabilities FROM nash_results ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    nash_probs = row[0] if row and row[0] else {}
    if isinstance(nash_probs, str): nash_probs = json.loads(nash_probs)

    # 3. Max Fitness from evolutionary_bot.py
    print("📥 3/5 - Retrieving Evolutionary Fitness...")
    cursor.execute("SELECT best_species, MAX(best_fitness) FROM evolutionary_results GROUP BY best_species")
    evo_stats = {row[0]: row[1] for row in cursor.fetchall()}

    # 4. Base Stats
    print("📥 4/5 - Retrieving Pokémon Stats...")
    cursor.execute('SELECT "name", "stats" FROM "pokemons"')
    pokemon_rows = cursor.fetchall()
    stats_db = {}
    for name, stat_data in pokemon_rows:
        if isinstance(stat_data, str):
            stat_data = json.loads(stat_data)
        stats_db[name] = stat_data or {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}

    # 5. Retrieving 1v1 battles as target Dataset (Limited to 200,000 for RAM)
    print("📥 5/5 - Retrieving Training Dataset (1v1 battles)...")
    cursor.execute("SELECT p1_name, p2_name, winner FROM tournament_results WHERE winner IN ('p1', 'p2') ORDER BY RANDOM() LIMIT 200000")
    matches = cursor.fetchall()

    cursor.close()
    conn.close()

    return winrates, nash_probs, evo_stats, stats_db, matches

def build_features(p1, p2, winrates, nash_probs, evo_stats, stats_db):
    """Calculates Pokémon 1's advantages over Pokémon 2"""
    s1 = stats_db.get(p1, {})
    s2 = stats_db.get(p2, {})
    
    return {
        'diff_global_wr': winrates.get(p1, 0.5) - winrates.get(p2, 0.5),
        'diff_nash': nash_probs.get(p1, 0.0) - nash_probs.get(p2, 0.0),
        'diff_evo': evo_stats.get(p1, 0.0) - evo_stats.get(p2, 0.0),
        'diff_bst': sum(s1.values()) - sum(s2.values()),
        'diff_spe': s1.get('spe', 0) - s2.get('spe', 0), # Speed Advantage
        'diff_atk': s1.get('atk', 0) - s2.get('atk', 0), # Physical Advantage
        'diff_def': s1.get('def', 0) - s2.get('def', 0)  # Defensive Advantage
    }

def main():
    while True:
        print("\n🚀 Starting Feature Engineering for 1v1 AI...")
        winrates, nash_probs, evo_stats, stats_db, matches = fetch_data()

        if not matches:
            print("❌ No 1v1 match found. Let bot.py run!")
            time.sleep(60)
            continue

        dataset = []
        for p1, p2, winner in matches:
            features = build_features(p1, p2, winrates, nash_probs, evo_stats, stats_db)
            # Target: 1 if P1 wins, 0 if P2 wins
            features['target'] = 1 if winner == 'p1' else 0
            features['p1_name'] = p1
            features['p2_name'] = p2
            dataset.append(features)

        df = pd.DataFrame(dataset)
        
        # Splitting while keeping names for results
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        
        X_train = df_train.drop(columns=['target', 'p1_name', 'p2_name'])
        y_train = df_train['target']
        
        X_test = df_test.drop(columns=['target', 'p1_name', 'p2_name'])
        y_test = df_test['target']

        print(f"\n🧠 Training AI (RandomForest) on {len(df)} 1v1 matches...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 Model accuracy on test set: {acc * 100:.2f}%")
        print("\n📊 Feature importance in decision making:")
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        for feat, imp in feature_importances.items():
            print(f"   - {feat}: {imp * 100:.1f}%")

        model_filename = "ai_1v1_model.pkl"
        joblib.dump(model, model_filename)
        print(f"\n💾 AI model locally saved as '{model_filename}'")
        
        # --- Exporting successes/failures ---
        results_df = df_test[['p1_name', 'p2_name', 'target']].copy()
        results_df['prediction'] = y_pred
        results_df['is_correct'] = results_df['target'] == results_df['prediction']
        
        save_test_results_to_db(results_df)
        
        print("\n⏳ Waiting 10 minutes before the next automatic retraining...")
        time.sleep(600)

if __name__ == "__main__":
    main()