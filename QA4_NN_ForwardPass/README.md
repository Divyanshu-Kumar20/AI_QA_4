# QA-4 (Unit III) — Neural Network Forward Pass (Computation Only)

This mini project implements **only the forward pass** of a simple neural network:

**Input (3 neurons) → Hidden (4 neurons, ReLU) → Output (2 neurons, Sigmoid)**

✅ Demonstrates:
- How inputs move through layers
- How weights & biases are applied
- How activation functions are used
- How final output is computed

❌ Not included (as per QA rules):
- Training
- Backpropagation
- Loss / optimization

---

## Folder Structure

```
QA4_NN_ForwardPass/
  main.py
  requirements.txt
  outputs/
    network_forward_pass.png
    input_values.png
    hidden_activations.png
    output_values.png
```

---

## How to Run

### 1) Install Python packages
```bash
pip install -r requirements.txt
```

### 2) Run program
```bash
python main.py
```

---

## Output Images (MANDATORY)

After running, the program automatically saves images inside `outputs/`:
- `network_forward_pass.png` (flow diagram with values)
- `input_values.png` (bar plot)
- `hidden_activations.png` (bar plot)
- `output_values.png` (bar plot)

---

## GitHub Submission Tip
Commit the code **and** the `outputs/` images:

```bash
git add .
git commit -m "QA-4 forward pass implementation with outputs"
git push
```
