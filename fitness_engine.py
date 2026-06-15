import random
import numpy as np
from typing import Dict, List, Tuple

"""
1. EXPECTED CHROMOSOME STRUCTURE BY THE ENGINE
Each individual in the population must exactly follow this structure:

individual = {
    "species_id": "mewtwo",
    "bst": 680,
    "base_stats": {"hp": 106, "atk": 110, "def": 90, "spa": 154, "spd": 90, "spe": 130},
    "moves": ["psychic", "blizzard", "recover", "amnesia"],
    "raw_wins": 0.0,
    "fitness": 0.0
}
"""

class FitnessEngine:
    def __init__(self, moves_db: Dict[str, dict], risk_aversion_factor: float = 0.9):
        """
        Heuristic 1v1 simulation engine for genetic algorithm.
        
        :param moves_db: Dictionary containing move data (power, accuracy).
        :param risk_aversion_factor: Multiplier to penalize moves with < 100% accuracy.
        """
        self.moves_db = moves_db
        self.risk_aversion = risk_aversion_factor

    def get_best_expected_damage(self, attacker: dict, defender: dict) -> float:
        """
        Calculates maximum damage of the attacker against the defender using
        the true proportional damage calculation formula (Atk vs Def / SpA vs SpD).
        """
        # Battle stats at level 100
        atk_stats = {k: v * 2 + 5 for k, v in attacker["base_stats"].items() if k != "hp"}
        def_stats = {k: v * 2 + 5 for k, v in defender["base_stats"].items() if k != "hp"}
        
        max_dmg = 0.0
        for move in attacker["moves"]:
            move_data = self.moves_db.get(move, {})
            power = move_data.get("power", 0)
            if power == 0:
                continue
                
            accuracy = move_data.get("accuracy", 100)
            if accuracy is True or not isinstance(accuracy, (int, float)):
                accuracy = 100.0
                
            category = move_data.get("category", "Physical")
            
            # Application of Physical / Special split
            if category == "Physical":
                off_stat = atk_stats["atk"]
                def_stat = def_stats["def"]
            else:
                off_stat = atk_stats["spa"]
                def_stat = def_stats["spd"]
                
            # Simplified mathematical Pokémon formula (Damage * Acc)
            base_dmg = ((power * (off_stat / def_stat)) / 5.0) + 2.0
            expected_dmg = base_dmg * (accuracy / 100.0)
            
            # Risk aversion for inaccurate moves
            if accuracy < 100.0:
                expected_dmg *= self.risk_aversion
            
            if expected_dmg > max_dmg:
                max_dmg = expected_dmg
                
        return max_dmg

    def simulate_battle(self, p1: dict, p2: dict) -> Tuple[float, float]:
        """
        Battle loop over multiple turns (up to 5). 
        Includes passive Anti-Stall: penalized victory if Time-To-Kill (TTK) is too long.
        """
        # Maximum HP calculation (Level 100)
        hp1_max = p1["base_stats"]["hp"] * 2 + 110
        hp2_max = p2["base_stats"]["hp"] * 2 + 110
        hp1, hp2 = hp1_max, hp2_max
        
        spe1 = p1["base_stats"]["spe"]
        spe2 = p2["base_stats"]["spe"]
        
        dmg1 = self.get_best_expected_damage(p1, p2)
        dmg2 = self.get_best_expected_damage(p2, p1)
        
        if dmg1 == 0 and dmg2 == 0:
            return 0.5, 0.5  # Absolute tie (2 Pokémon with harmless moves)
            
        p1_faster = spe1 > spe2
        if spe1 == spe2:
            p1_faster = random.choice([True, False])
            
        turns = 0
        max_turns = 5
        
        while hp1 > 0 and hp2 > 0 and turns < max_turns:
            turns += 1
            if p1_faster:
                hp2 -= dmg1
                if hp2 <= 0: break
                hp1 -= dmg2
            else:
                hp1 -= dmg2
                if hp1 <= 0: break
                hp2 -= dmg1
                
        # Linear Anti-Stall penalty calculation (-0.25 at T5)
        penalty = (turns - 4) * 0.25 if turns > 4 else 0.0
        
        if hp1 > 0 and hp2 <= 0:
            return max(0.0, 1.0 - penalty), 0.0
        elif hp2 > 0 and hp1 <= 0:
            return 0.0, max(0.0, 1.0 - penalty)
        else:
            # Both survived 5 turns (Stall), victory goes to the one with highest % HP remaining
            pct1, pct2 = hp1 / hp1_max, hp2 / hp2_max
            if pct1 > pct2:
                return max(0.0, 0.8 - penalty), 0.0  # Bigger penalty as there's no real KO
            elif pct2 > pct1:
                return 0.0, max(0.0, 0.8 - penalty)
            return 0.5, 0.5

    def apply_bulk_penalty(self, bst: int, raw_wins: float) -> float:
        """
        4. BULK PENALTY
        Formula: Final_Fitness = Win_Score * (Pokemon_BST / 600)
        """
        # We use max(1, raw_wins) or keep negatives if massively OHKO'd
        return raw_wins * (bst / 600.0)

    def evaluate_population(self, population: List[dict], matches_per_ind: int = 10) -> None:
        """
        Orchestrates matches and updates the 'fitness' of each individual in-place.
        """
        # Phase 1: Battles
        for p1 in population:
            opponents = random.sample(population, min(matches_per_ind, len(population)))
            for p2 in opponents:
                if p1 is p2: continue
                s1, s2 = self.simulate_battle(p1, p2)
                p1["raw_wins"] += s1
                p2["raw_wins"] += s2

        # Phase 2: Final calculation with Bulk Penalty
        for ind in population:
            ind["fitness"] = self.apply_bulk_penalty(ind["bst"], ind["raw_wins"])