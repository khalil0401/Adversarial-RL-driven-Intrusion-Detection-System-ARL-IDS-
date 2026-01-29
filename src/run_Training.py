# 4. Run Training
if __name__ == "__main__":
    try:
        print(f"Starting training for {args.episodes} episodes with decay {args.epsilon_decay}...")
        train(args)
    except FileNotFoundError:
        print("\n[ERROR] Dataset not found! Please upload the dataset and update 'args.data_path' in the previous cell.")