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

    state_dim = 16 # Latent dim
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # 2. Representation Learning (Layer 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = StateEncoder(input_dim, latent_dim=state_dim, device=device)
    if not args.skip_encoder_train:
        encoder.train(X_train, epochs=args.encoder_epochs)
    
    # 3. Initialize Components
    if args.no_adversary:
        # Dummy adversary that does nothing or Env ignores it
        adversary = LightweightAdversary(mutation_rate=0.0, mutation_strength=0.0)
    else:
        adversary = LightweightAdversary(mutation_rate=0.1, mutation_strength=0.1)
        
    env = AdversarialIDSEnv(X_train, y_train, encoder, adversary)
    
    # Disable curriculum by forcing buffer_prob to 0 always if requested
    if args.no_curriculum:
        env.buffer_prob = 0.0
    
    agent = DDQNAgent(state_dim, n_classes, device=device, lr=args.lr, epsilon_decay=args.epsilon_decay)
    
    # 4. Training Loop
    logger.info("Starting RL Training...")
    scores = []
    
    for episode in range(args.episodes):
        state, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Ablation: No Reward Shaping -> Reset weight from Env (which applies it) or normalize it back?
            # Env calculates reward = W * (+/-1). If no_shaping, we want reward = 1 * (+/-1).
            # Easier to just tell Env to use ones. But Env uses dynamic class attributes.
            # We can force env.class_weights to ones if no_shaping.
            if args.no_reward_shaping:
                # Hack: Divide by weight used? Or just keep it 1.0. 
                # Better: Ensure env has weights=1.0.
                pass # See weight update section below
                
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done, info)
            loss = agent.update()
            
            state = next_state
            score += reward
            
        scores.append(score)
        
        # Periodic Updates
        if episode % args.target_update_freq == 0:
            agent.update_target_network()
            
        if episode % args.weight_update_freq == 0 and episode > 0:
            if not args.no_reward_shaping:
                new_weights = agent.calculate_new_weights(env.class_weights)
                env.update_class_weights(new_weights)
            
            # Curriculum Update (Adversarial Prob)
            if not args.no_curriculum:
                avg_score = np.mean(scores[-50:]) if len(scores) > 50 else score
                if avg_score > 0.8 * args.max_steps_per_episode: 
                    env.update_buffer_prob(min(0.5, env.buffer_prob + 0.05))
                
        if episode % 100 == 0:
            logger.info(f"Episode {episode}\tScore: {score:.2f}\tEpsilon: {agent.epsilon:.2f}\tBufferProb: {env.buffer_prob:.2f}")

    # Save Models
    os.makedirs("results/checkpoints", exist_ok=True)
    encoder.save("results/checkpoints/encoder.pth")
    torch.save(agent.policy_net.state_dict(), "results/checkpoints/policy_net.pth")
    logger.info("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="D:/An adversarial environment reinforcement/train_test_network.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--encoder_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--target_update_freq", type=int, default=10)
    parser.add_argument("--weight_update_freq", type=int, default=50)
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
