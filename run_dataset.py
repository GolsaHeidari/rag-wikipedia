print("🔥 RUN DATASET STARTED")

from src.dataset import load_and_save_dataset

def main():
    print("📦 Starting dataset pipeline...")
    load_and_save_dataset()
    print("🎉 Done!")

if __name__ == "__main__":
    main()