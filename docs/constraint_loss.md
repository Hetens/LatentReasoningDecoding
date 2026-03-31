# Sudoku Constraint-Aware Auxiliary Loss

To improve the Transformer's ability to solve Sudoku puzzles, we add a differentiable "constraint loss" that penalizes the model when its predicted probabilities violate Sudoku rules (duplicate digits in rows, columns, or 3x3 boxes).

## Motivation

Standard Cross-Entropy loss only teaches the model to match the ground-truth answer. It doesn't explicitly penalize the model for being "illogical" (e.g., predicting two '5's in the same row) as long as it gets the "correct" position right. 

The constraint loss provides a structural signal: **"Whatever you predict, make sure it's a valid Sudoku layout."**

## How it Works

The loss is computed on the **probabilities** (after Softmax) for each of the 81 answer cells.

### 1. Identify Units
We define 27 "units" of 9 cells each:
- 9 Rows
- 9 Columns
- 9 3x3 Boxes

### 2. Penalty Formula
For each unit and each digit $d \in \{1 \dots 9\}$, we compute:

$$\text{Penalty}_d = \sum_{i=1}^9 (p_{i,d})^2 - \max_{i=1}^9 (p_{i,d})^2$$

Where $p_{i,d}$ is the probability of cell $i$ in that unit being digit $d$.

- **If only one cell** has a high probability for digit $d$, the sum of squares is roughly $1^2 = 1$ and the max square is $1^2 = 1$. The difference is **0**.
- **If two cells** both have 0.7 probability for digit $d$, the sum of squares is $0.7^2 + 0.7^2 = 0.98$, and the max square is $0.49$. The penalty is **0.49**.
- **If all cells** have low probability (spread out), the squares are small, resulting in a low penalty.

### 3. Total Loss
The final loss is:
$$\text{Total Loss} = \text{Cross-Entropy} + \lambda \cdot \text{Constraint Loss}$$

We use $\lambda = 0.1$ by default. This forces the model's "internal logic" to respect the uniqueness constraints of Sudoku during gradients, even before it has seen the full Ground Truth for a specific cell.
