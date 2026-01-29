import argparse
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def parse_logs(log_file):
    episodes = []
    def_scores = []
    atk_scores = []
    epsilons = []
    f1_scores = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    current_ep = 0
    
    for line in lines:
        # Parse Score Line
        # INFO:Train:Ep 100 | Def Score: -0.78 (Eps: 1.00) | Atk Score: 0.78 (Eps: 0.83)
        score_match = re.search(r"Ep (\d+) \| Def Score: ([-\d\.]+) \(Eps: ([\d\.]+)\) \| Atk Score: ([-\d\.]+)", line)
        if score_match:
            ep = int(score_match.group(1))
            d_score = float(score_match.group(2))
            eps = float(score_match.group(3))
            a_score = float(score_match.group(4))
            
            episodes.append(ep)
            def_scores.append(d_score)
            atk_scores.append(a_score)
            epsilons.append(eps)
            current_ep = ep
            
        # Parse F1 Line (Occurs after scores usually)
        # INFO:DDQN_Agent:Class F1 Scores: [0.1 0.2 ...]
        f1_match = re.search(r"Class F1 Scores: \[([\d\. ]+)\]", line)
        if f1_match:
            # Clean string and convert to list
            vals_str = f1_match.group(1).replace('\n', ' ')
            # Handle multiple spaces
            vals = [float(x) for x in vals_str.split() if x.strip()]
            f1_scores.append({'episode': current_ep, 'f1': vals})

    return {
        'episodes': episodes,
        'def_scores': def_scores,
        'atk_scores': atk_scores,
        'epsilons': epsilons,
        'f1_scores': f1_scores
    }

def plot_training(data, output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Competitive Scores
    plt.figure(figsize=(10, 6))
    plt.plot(data['episodes'], data['def_scores'], label='Defender Score', color='blue', alpha=0.7)
    plt.plot(data['episodes'], data['atk_scores'], label='Attacker Score', color='red', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Last 100)')
    plt.title('Competitive Training Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/competitive_scores.png")
    plt.close()
    
    # 2. Epsilon Decay
    plt.figure(figsize=(8, 4))
    plt.plot(data['episodes'], data['epsilons'], label='Epsilon', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon (Exploration Rate)')
    plt.title('Exploration Decay')
    plt.grid(True)
    plt.savefig(f"{output_dir}/epsilon_decay.png")
    plt.close()
    
    # 3. Class F1 Heatmap or Line Plot
    if data['f1_scores']:
        f1_data = [] # List of lists
        eps = []
        for item in data['f1_scores']:
            if len(item['f1']) > 0:
                f1_data.append(item['f1'])
                eps.append(item['episode'])
        
        f1_array = np.array(f1_data) # Shape (T, N_classes)
        
        if f1_array.size > 0:
            plt.figure(figsize=(12, 8))
            # Plot top 5 classes or all if few
            for i in range(f1_array.shape[1]):
                plt.plot(eps, f1_array[:, i], label=f'Class {i}', alpha=0.5)
            
            plt.xlabel('Episode')
            plt.ylabel('F1 Score')
            plt.title('Class-wise F1 Score Evolution')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/class_f1_history.png")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Path to text log file")
    args = parser.parse_args()
    
    print(f"Parsing logs from {args.log_file}...")
    data = parse_logs(args.log_file)
    print(f"Found {len(data['episodes'])} data points.")
    
    plot_training(data)
    print("Plots saved to results/plots/")
