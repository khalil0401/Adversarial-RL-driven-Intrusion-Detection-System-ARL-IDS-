import argparse
import logging
import numpy as np
import torch
import os
import json
from src.data.ton_iot_loader import TonIoTLoader
from src.representation.encoder import StateEncoder
from src.envs.adversarial_ids_env import AdversarialIDSEnv
from src.agents.ddqn_agent import DDQNAgent
from src.adversary.lightweight_generator import LightweightAdversary

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Train")

def train(args):
    # Set Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 1. Load Data
    loader = TonIoTLoader(args.data_path, seed=args.seed)
    # Mocking loading for now if file not present, user should provide correct path
    if not os.path.exists(args.data_path):
        logger.warning(f"Data path {args.data_path} not found. Creating dummy data for structure verification.")
        X_train = np.random.rand(100, 38)
        y_train = np.random.randint(0, 10, 100)
        X_test = np.random.rand(20, 38)
        y_test = np.random.randint(0, 10, 20)
        loader.classes_ = list(range(10)) # Mock 10 classes
    else:
        X_train, X_test, y_train, y_test = loader.load_and_process()

    state_dim = 64 # Latent dim
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # 2. Representation Learning (Layer 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = StateEncoder(input_dim, latent_dim=state_dim, n_classes=n_classes, device=device)
    if not args.skip_encoder_train:
        encoder.train(X_train, y_train, epochs=args.encoder_epochs)
    
    # 3. Initialize Components
    # 3. Initialize Components
    # Adversary Agent (RL)
    from src.agents.attacker_agent import AttackerAgent
    # Input dim for attacker is raw features (38)
    attacker = AttackerAgent(input_dim=input_dim, action_dim=input_dim*2, device=device)
    
    # Environment (Now just a container for data & transitions, adversary logic moved to agent)
    # We pass None as adversary to Env because we handle mutation manually in the loop
    env = AdversarialIDSEnv(X_train, y_train, encoder, adversary=None) 
    
    # Disable internal curriculum buffer if we are using active Attacker Agent (to avoid double shifting)
    # Or keep it as "Hard Sample Mining"? Let's disable Env buffer to isolate Attacker Agent efficacy.
    env.buffer_prob = 0.0
    
    # Defender Agent
    agent = DDQNAgent(state_dim, n_classes, device=device, lr=args.lr, epsilon_decay=args.epsilon_decay)
    
    # 4. Competitive Training Loop
    logger.info("Starting Competitive RL Training (Attacker vs Defender)...")
    scores_def = []
    scores_atk = []
    
    for episode in range(args.episodes):
        # 1. Reset Env (Get clean sample)
        # Note: We need raw sample for Attacker, Env.reset() returns encoded state.
        # We need to peek at env.current_sample
        state_def_clean, _ = env.reset() 
        raw_sample = env.current_sample # Get raw features (38,)
        
        # 2. Attacker Turn
        if not args.no_adversary:
            action_atk = attacker.select_action(raw_sample)
            perturbation = attacker.get_perturbation(action_atk)
        else:
            # No Adversary: Zero perturbation (Clean Training)
            perturbation = np.zeros_like(raw_sample)
            action_atk = 0 # Dummy action
        
        # Apply perturbation
        raw_sample_adv = raw_sample + perturbation
        raw_sample_adv = np.clip(raw_sample_adv, 0.0, 1.0) # Ensure valid range
        
        # 3. Defender Turn
        # Encode adversary sample
        state_def_adv = encoder.get_latent(raw_sample_adv).flatten()
        action_def = agent.select_action(state_def_adv)
        
        # 4. Step Logic (Get Reward)
        # We manually calculate reward derived from Env logic
        true_class = env.current_label
        weight = env.class_weights[true_class]
        
        reward_def = 0
        if action_def == true_class:
            reward_def = 1.0 * weight
        else:
            reward_def = -1.0 * weight
            
        # Zero-Sum Reward for Attacker (Clipped to avoid explosion)
        reward_atk = -reward_def 
        
        # 5. Store Transitions
        # Defender: (State_Adv, Action, Reward, Next_State(Terminal), Done)
        agent.store_transition(state_def_adv, action_def, reward_def, state_def_adv, True, {"true_class": true_class})
        
        # Attacker: (Raw_State, Action, Reward, Next_Raw_State(Terminal), Done)
        # Only store and update if Adversary is active
        if not args.no_adversary:
            attacker.store_transition(raw_sample, action_atk, reward_atk, raw_sample_adv, True)
        
        # 6. Updates
        loss_def = agent.update()
        
        if not args.no_adversary:
            loss_atk = attacker.update()
        
        scores_def.append(reward_def)
        if not args.no_adversary:
            scores_atk.append(reward_atk)
        
        # Periodic Updates
        if episode % args.target_update_freq == 0:
            agent.update_target_network()
            attacker.update_target_network()
            
        if episode % args.weight_update_freq == 0 and episode > 0:
            if not args.no_reward_shaping:
                new_weights = agent.calculate_new_weights(env.class_weights)
                env.update_class_weights(new_weights)
                
        if episode % 100 == 0:
            # Calculate win rates
            recent_def = np.mean(scores_def[-100:])
            recent_atk = np.mean(scores_atk[-100:])
            logger.info(f"Ep {episode} | Def Score: {recent_def:.2f} (Eps: {agent.epsilon:.2f}) | Atk Score: {recent_atk:.2f} (Eps: {attacker.epsilon:.2f})")

    # Save Models
    os.makedirs("results/checkpoints", exist_ok=True)
    encoder.save("results/checkpoints/encoder.pth")
    torch.save(agent.policy_net.state_dict(), "results/checkpoints/policy_net.pth")
    torch.save(attacker.policy_net.state_dict(), "results/checkpoints/attacker_net.pth")
    logger.info("Competitive Training Complete.")

if __name__ == "__main__":
    from src.config import Config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=Config.get_data_path())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=Config.EPISODES)
    parser.add_argument("--encoder_epochs", type=int, default=Config.ENCODER_EPOCHS)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--epsilon_decay", type=float, default=Config.EPSILON_DECAY)
    parser.add_argument("--target_update_freq", type=int, default=Config.TARGET_UPDATE_FREQ)
    parser.add_argument("--weight_update_freq", type=int, default=Config.WEIGHT_UPDATE_FREQ)
    parser.add_argument("--skip_encoder_train", action="store_true")
    # Hack for undefined max_steps_per_episode in env
    # In my env implementation, step returns Terminated=True immediately? No, wait.
    # Review env step logic: "terminated = True" IS set immediately in my code.
    # Ah, Classification as RL usually treats one sample as one episode OR one episode = one epoch over data.
    # My Env.reset() picks ONE sample. Step predicts. Done. 
    # So max_score is roughly just max_reward (e.g. 10 * weight_max).
    parser.add_argument("--max_steps_per_episode", type=int, default=1) 
    
    # Ablation Flags
    parser.add_argument("--no_reward_shaping", action="store_true", help="Disable dynamic reward weighting")
    parser.add_argument("--no_adversary", action="store_true", help="Disable adversarial generation (static dataset)")
    parser.add_argument("--no_curriculum", action="store_true", help="Disable curriculum learning (random sampling)")

    args = parser.parse_args()
    train(args)
