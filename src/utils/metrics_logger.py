"""
Training Metrics Logger for ARL-IDS

Tracks and logs training metrics for both defender and attacker agents,
providing real-time monitoring and balance detection.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np


class MetricsLogger:
    """
    Comprehensive metrics logger for competitive RL training.
    
    Tracks:
    - Defender/Attacker scores and losses
    - Per-class F1 scores and weights
    - Epsilon values (exploration)
    - Training balance indicators
    """
    
    def __init__(self, log_dir="results/logs", experiment_name=None):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (default: timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        self.metrics = {
            "defender_scores": [],
            "attacker_scores": [],
            "defender_losses": [],
            "attacker_losses": [],
            "defender_epsilon": [],
            "attacker_epsilon": [],
            "class_f1_scores": [],
            "class_weights": [],
            "episodes": [],
            "imbalance_warnings": []
        }
        
        self.current_episode = 0
        
    def log_step(self, episode, defender_score, attacker_score, 
                 defender_loss=None, attacker_loss=None,
                 defender_epsilon=None, attacker_epsilon=None,
                 class_f1=None, class_weights=None):
        """
        Log metrics for a single training step.
        
        Args:
            episode: Current episode number
            defender_score: Defender reward this episode
            attacker_score: Attacker reward this episode
            defender_loss: Defender Q-learning loss (optional)
            attacker_loss: Attacker Q-learning loss (optional)
            defender_epsilon: Defender exploration rate (optional)
            attacker_epsilon: Attacker exploration rate (optional)
            class_f1: Array of per-class F1 scores (optional)
            class_weights: Array of class weights (optional)
        """
        self.current_episode = episode
        self.metrics["episodes"].append(episode)
        self.metrics["defender_scores"].append(float(defender_score))
        self.metrics["attacker_scores"].append(float(attacker_score))
        
        if defender_loss is not None:
            self.metrics["defender_losses"].append(float(defender_loss))
        if attacker_loss is not None:
            self.metrics["attacker_losses"].append(float(attacker_loss))
        if defender_epsilon is not None:
            self.metrics["defender_epsilon"].append(float(defender_epsilon))
        if attacker_epsilon is not None:
            self.metrics["attacker_epsilon"].append(float(attacker_epsilon))
        if class_f1 is not None:
            self.metrics["class_f1_scores"].append(class_f1.tolist() if isinstance(class_f1, np.ndarray) else class_f1)
        if class_weights is not None:
            self.metrics["class_weights"].append(class_weights.tolist() if isinstance(class_weights, np.ndarray) else class_weights)
    
    def check_training_balance(self, window=100):
        """
        Check if training is balanced between defender and attacker.
        
        Returns alert if one agent is dominating (win rate > 80% or < 20%).
        
        Args:
            window: Number of recent episodes to check
            
        Returns:
            dict with balance status and recommendations
        """
        if len(self.metrics["defender_scores"]) < window:
            return {"balanced": True, "message": "Insufficient data"}
        
        recent_def = self.metrics["defender_scores"][-window:]
        recent_atk = self.metrics["attacker_scores"][-window:]
        
        # Calculate win rates (positive score = win)
        def_wins = sum(1 for s in recent_def if s > 0)
        def_win_rate = def_wins / window
        
        alert = {"balanced": True, "message": "Training balanced"}
        
        if def_win_rate > 0.8:
            alert = {
                "balanced": False,
                "message": f"⚠️  DEFENDER DOMINATING (Win rate: {def_win_rate:.1%})",
                "recommendation": "Consider: Increase attacker learning rate or decrease defender epsilon decay"
            }
            self.metrics["imbalance_warnings"].append({
                "episode": self.current_episode,
                "type": "defender_dominance",
                "win_rate": def_win_rate
            })
        elif def_win_rate < 0.2:
            alert = {
                "balanced": False,
                "message": f"⚠️  ATTACKER DOMINATING (Defender win rate: {def_win_rate:.1%})",
                "recommendation": "Consider: Increase defender learning rate or adjust reward shaping"
            }
            self.metrics["imbalance_warnings"].append({
                "episode": self.current_episode,
                "type": "attacker_dominance",
                "win_rate": def_win_rate
            })
        
        return alert
    
    def get_summary(self, window=100):
        """
        Get summary statistics for recent training.
        
        Args:
            window: Number of recent episodes
            
        Returns:
            dict with summary statistics
        """
        if len(self.metrics["episodes"]) == 0:
            return {"error": "No metrics logged"}
        
        recent_def = self.metrics["defender_scores"][-window:]
        recent_atk = self.metrics["attacker_scores"][-window:]
        
        summary = {
            "total_episodes": self.current_episode,
            "avg_defender_score": np.mean(recent_def),
            "avg_attacker_score": np.mean(recent_atk),
            "defender_win_rate": sum(1 for s in recent_def if s > 0) / len(recent_def),
        }
        
        if self.metrics["defender_epsilon"]:
            summary["current_epsilon_def"] = self.metrics["defender_epsilon"][-1]
        if self.metrics["attacker_epsilon"]:
            summary["current_epsilon_atk"] = self.metrics["attacker_epsilon"][-1]
        
        return summary
    
    def save(self, filename=None):
        """
        Save metrics to JSON file.
        
        Args:
            filename: Output filename (default: experiment_name.json)
        """
        if filename is None:
            filename = f"{self.experiment_name}_metrics.json"
        
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"[SUCCESS] Metrics saved to: {filepath}")
        return filepath
    
    def save_csv(self, filename=None):
        """
        Save metrics to CSV file for easy analysis.
        
        Args:
            filename: Output filename (default: experiment_name.csv)
        """
        if filename is None:
            filename = f"{self.experiment_name}_metrics.csv"
        
        filepath = self.log_dir / filename
        
        # Create CSV data
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ["episode", "defender_score", "attacker_score"]
            if self.metrics["defender_losses"]:
                header.extend(["defender_loss", "attacker_loss"])
            if self.metrics["defender_epsilon"]:
                header.extend(["defender_epsilon", "attacker_epsilon"])
            writer.writerow(header)
            
            # Data
            n = len(self.metrics["episodes"])
            for i in range(n):
                row = [
                    self.metrics["episodes"][i],
                    self.metrics["defender_scores"][i],
                    self.metrics["attacker_scores"][i]
                ]
                if self.metrics["defender_losses"]:
                    row.extend([
                        self.metrics["defender_losses"][i] if i < len(self.metrics["defender_losses"]) else "",
                        self.metrics["attacker_losses"][i] if i < len(self.metrics["attacker_losses"]) else ""
                    ])
                if self.metrics["defender_epsilon"]:
                    row.extend([
                        self.metrics["defender_epsilon"][i] if i < len(self.metrics["defender_epsilon"]) else "",
                        self.metrics["attacker_epsilon"][i] if i < len(self.metrics["attacker_epsilon"]) else ""
                    ])
                writer.writerow(row)
        
        print(f"[SUCCESS] CSV saved to: {filepath}")
        return filepath
