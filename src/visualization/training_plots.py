"""
Training Visualization Utilities for ARL-IDS

Generates publication-quality plots for training analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_training_curves(metrics_file, output_dir="results/plots", show=False):
    """
    Generate comprehensive training visualization from metrics file.
    
    Args:
        metrics_file: Path to metrics JSON file
        output_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    episodes = np.array(metrics["episodes"])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("ARL-IDS Training Dynamics", fontsize=16, fontweight='bold')
    
    # 1. Defender vs Attacker Scores
    ax = axes[0, 0]
    if len(metrics["defender_scores"]) > 0:
        # Moving average for smoothing
        window = min(100, len(episodes) // 10)
        def_scores = np.array(metrics["defender_scores"])
        atk_scores = np.array(metrics["attacker_scores"])
        
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        if len(def_scores) > window:
            def_smooth = moving_average(def_scores, window)
            atk_smooth = moving_average(atk_scores, window)
            episodes_smooth = episodes[window-1:]
            
            ax.plot(episodes_smooth, def_smooth, label='Defender (smoothed)', color='#2ecc71', linewidth=2)
            ax.plot(episodes_smooth, atk_smooth, label='Attacker (smoothed)', color='#e74c3c', linewidth=2)
        
        ax.plot(episodes, def_scores, alpha=0.1, color='#2ecc71')
        ax.plot(episodes, atk_scores, alpha=0.1, color='#e74c3c')
        
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Competitive Learning Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Epsilon Decay
    ax = axes[0, 1]
    if metrics.get("defender_epsilon"):
        ax.plot(episodes, metrics["defender_epsilon"], label='Defender ε', color='#3498db', linewidth=2)
        ax.plot(episodes, metrics["attacker_epsilon"], label='Attacker ε', color='#9b59b6', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon (Exploration Rate)')
        ax.set_title('Exploration Decay')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No epsilon data', ha='center', va='center', transform=ax.transAxes)
    
    # 3. Loss Curves
    ax = axes[1, 0]
    if metrics.get("defender_losses"):
        def_losses = np.array(metrics["defender_losses"])
        atk_losses = np.array(metrics["attacker_losses"])
        
        # Moving average
        window = min(50, len(def_losses) // 10)
        if len(def_losses) > window:
            def_loss_smooth = moving_average(def_losses, window)
            atk_loss_smooth = moving_average(atk_losses, window)
            loss_episodes = np.array(episodes[:len(def_losses)])[window-1:]
            
            ax.plot(loss_episodes, def_loss_smooth, label='Defender Loss', color='#2ecc71', linewidth=2)
            ax.plot(loss_episodes, atk_loss_smooth, label='Attacker Loss', color='#e74c3c', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Q-Learning Loss')
        ax.set_title('Training Loss Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax.transAxes)
    
    # 4. Win Rate Evolution
    ax = axes[1, 1]
    if len(metrics["defender_scores"]) > 100:
        def_scores = np.array(metrics["defender_scores"])
        window = 100
        win_rates = []
        win_rate_episodes = []
        
        for i in range(window, len(def_scores)):
            recent = def_scores[i-window:i]
            win_rate = sum(1 for s in recent if s > 0) / window
            win_rates.append(win_rate)
            win_rate_episodes.append(episodes[i])
        
        ax.plot(win_rate_episodes, win_rates, color='#3498db', linewidth=2)
        ax.axhline(y=0.5, color='green', linestyle='--', label='Balanced (50%)')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Defender Dominating')
        ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Attacker Dominating')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Defender Win Rate (100-ep window)')
        ax.set_title('Training Balance Monitor')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient data (need 100+ episodes)', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / "training_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_file


def plot_class_f1_heatmap(metrics_file, output_dir="results/plots", show=False):
    """
    Generate heatmap of per-class F1 scores over time.
    
    Args:
        metrics_file: Path to metrics JSON file
        output_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if not metrics.get("class_f1_scores"):
        print("⚠️  No class F1 scores in metrics file")
        return None
    
    # Convert to matrix
    f1_matrix = np.array(metrics["class_f1_scores"])  # Shape: (n_updates, n_classes)
    
    if f1_matrix.size == 0:
        print("⚠️  Empty F1 score data")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    im = ax.imshow(f1_matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    # Labels
    n_classes = f1_matrix.shape[1]
    ax.set_xlabel('Weight Update Iteration', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([f'Class {i}' for i in range(n_classes)])
    ax.set_title('Per-Class F1 Score Evolution', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('F1 Score', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    output_file = output_dir / "class_f1_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ F1 heatmap saved to: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_file


def generate_all_plots(metrics_file, output_dir="results/plots"):
    """
    Generate all training visualizations.
    
    Args:
        metrics_file: Path to metrics JSON file
        output_dir: Directory to save plots
    """
    print(f"\n📊 Generating training visualizations...")
    print(f"   Metrics file: {metrics_file}")
    print(f"   Output directory: {output_dir}\n")
    
    plot_training_curves(metrics_file, output_dir)
    plot_class_f1_heatmap(metrics_file, output_dir)
    
    print(f"\n✅ All visualizations generated in: {output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]
        generate_all_plots(metrics_file)
    else:
        print("Usage: python training_plots.py <metrics_file.json>")
