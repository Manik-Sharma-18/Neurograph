# main.py

from train.train_context import train_context
from config import load_config
import matplotlib.pyplot as plt
import os

def main():
    cfg = load_config("config/default.yaml")

    print("\n🚀 Starting NeuroGraph Training...\n")
    loss_log = train_context()

    # Create logs folder if needed
    os.makedirs(cfg.get("log_path", "logs/"), exist_ok=True)

    # Plot convergence
    plt.plot(loss_log)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Convergence")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(cfg.get("log_path", "logs/"), "loss_curve.png")
    plt.savefig(out_path)
    print(f"\n📉 Saved convergence plot to {out_path}")

if __name__ == "__main__":
    main()
