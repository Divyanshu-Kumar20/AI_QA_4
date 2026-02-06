"""
QA-4: Neural Network Forward Pass (Computation Flow Only)
- Input layer -> Hidden layer (ReLU) -> Output layer (Sigmoid)
- Hardcoded weights/biases
- Prints step-by-step calculations + saves output images to ./outputs

Run:
  python main.py
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Activations ----------
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

# ---------- Forward Pass ----------
def forward_pass(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray):
    # Hidden layer
    z1 = W1 @ x + b1
    a1 = relu(z1)

    # Output layer
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)

    return z1, a1, z2, a2

def pretty(name: str, arr: np.ndarray):
    print(f"\n{name} (shape={arr.shape}):\n{arr}")

def save_bar_plot(values: np.ndarray, title: str, filepath: Path):
    plt.figure()
    plt.title(title)
    plt.bar(range(len(values)), values)
    plt.xlabel("Neuron index")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()

def save_network_diagram(x, z1, a1, z2, a2, filepath: Path):
    # Simple diagram (no external libs): draw circles + text
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.set_axis_off()

    # positions
    x_col = 0.1
    h_col = 0.5
    o_col = 0.9

    # vertical spacing
    def positions(n, top=0.8, bottom=0.2):
        if n == 1:
            return [(top+bottom)/2]
        return np.linspace(top, bottom, n)

    in_y = positions(len(x))
    h_y  = positions(len(a1))
    o_y  = positions(len(a2))

    # draw nodes
    def node(col, y, label):
        circ = plt.Circle((col, y), 0.035, fill=False)
        ax.add_patch(circ)
        ax.text(col, y, label, ha="center", va="center", fontsize=8)

    for i, y in enumerate(in_y):
        node(x_col, y, f"x{i}={x[i]:.2f}")

    for j, y in enumerate(h_y):
        node(h_col, y, f"z1{j}={z1[j]:.2f}\na1{j}={a1[j]:.2f}")

    for k, y in enumerate(o_y):
        node(o_col, y, f"z2{k}={z2[k]:.2f}\na2{k}={a2[k]:.2f}")

    # connect columns (fully)
    for y1 in in_y:
        for y2 in h_y:
            ax.plot([x_col+0.035, h_col-0.035], [y1, y2], linewidth=0.8)

    for y1 in h_y:
        for y2 in o_y:
            ax.plot([h_col+0.035, o_col-0.035], [y1, y2], linewidth=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title("Forward Pass Flow (values shown)")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()

def main():
    # ---------- Input ----------
    # Example input vector (3 features)
    x = np.array([0.50, -1.20, 0.30])

    # ---------- Hardcoded Weights/Biases ----------
    # Hidden layer: 4 neurons, input size = 3  => W1 shape (4,3), b1 shape (4,)
    W1 = np.array([
        [ 0.20, -0.10,  0.40],
        [-0.70,  0.30,  0.10],
        [ 0.50,  0.80, -0.60],
        [ 0.10, -0.40,  0.20],
    ])
    b1 = np.array([0.10, -0.20, 0.05, 0.00])

    # Output layer: 2 neurons, hidden size = 4 => W2 shape (2,4), b2 shape (2,)
    W2 = np.array([
        [ 0.30, -0.20,  0.10,  0.50],
        [-0.40,  0.60, -0.10,  0.20],
    ])
    b2 = np.array([0.00, 0.10])

    print("\n=== QA-4: Neural Network Forward Pass (No Training) ===")
    pretty("Input x", x)
    pretty("W1 (hidden weights)", W1)
    pretty("b1 (hidden bias)", b1)

    # Forward
    z1, a1, z2, a2 = forward_pass(x, W1, b1, W2, b2)

    print("\n--- Hidden Layer Computation ---")
    print("z1 = W1 @ x + b1")
    pretty("z1", z1)
    print("a1 = ReLU(z1)")
    pretty("a1", a1)

    pretty("W2 (output weights)", W2)
    pretty("b2 (output bias)", b2)

    print("\n--- Output Layer Computation ---")
    print("z2 = W2 @ a1 + b2")
    pretty("z2", z2)
    print("a2 = Sigmoid(z2)  (final output probabilities)")
    pretty("a2 (final output)", a2)

    # ---------- Save Output Images (Mandatory) ----------
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)

    # 1) Network diagram with values
    save_network_diagram(x, z1, a1, z2, a2, outputs/"network_forward_pass.png")

    # 2) Bar plots for activations
    save_bar_plot(x,  "Input Layer Values", outputs/"input_values.png")
    save_bar_plot(a1, "Hidden Layer Activations (ReLU)", outputs/"hidden_activations.png")
    save_bar_plot(a2, "Output Layer (Sigmoid)", outputs/"output_values.png")

    print("\nSaved images to ./outputs/")
    print(" - network_forward_pass.png")
    print(" - input_values.png")
    print(" - hidden_activations.png")
    print(" - output_values.png")

if __name__ == "__main__":
    main()
