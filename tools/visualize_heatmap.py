import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def visualize(pred_file, true_file=None):
    pred = np.loadtxt(pred_file, delimiter=",")
    
    plt.figure(figsize=(10, 4))

    # Predicted
    plt.subplot(1, 2, 1)
    plt.imshow(pred, cmap="hot")
    plt.title("Predicted IR Drop")
    plt.colorbar()

    if true_file and os.path.exists(true_file):
        true = np.loadtxt(true_file, delimiter=",")
        # Ground Truth
        plt.subplot(1, 2, 2)
        plt.imshow(true, cmap="hot")
        plt.title("Ground Truth IR Drop")
        plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred", required=True, help="Path to predicted_ir_drop_map_*.csv")
    parser.add_argument("-true", required=False, help="Path to ground truth ir_drop_map_*.csv")
    args = parser.parse_args()

    visualize(args.pred, args.true)

