# DeepCFR: PyTorch Implementation for Kuhn Poker

A PyTorch implementation of **Deep Counterfactual Regret Minimization (Deep CFR)** applied to **Kuhn Poker**, based on the paper: [Deep CFR](https://arxiv.org/abs/1811.00164).

This project demonstrates how neural networks can approximate **regret** and **average strategies** in imperfect-information games.

---

## Features

- **Kuhn Poker environment** using PettingZoo (`kuhn.py`)
- **DeepCFR module** (`dcfr.py`) with:
  - `RegretNet` to approximate instantaneous regrets
  - `PolicyNet` to approximate the average strategy
- Infoset encoding (`encoding.py`) to convert cards and history into network inputs
- Simple **evaluation script** (`eval.py`) to compute win rates
- Game tree representation (`kuhn_state.py`) for traversals and CFR computations

---

## File Overview

| File | Description |
|------|-------------|
| `kuhn.py` | PettingZoo environment for Kuhn Poker |
| `kuhn_state.py` | State representation for CFR traversals |
| `models.py` | `RegretNet` and `PolicyNet` neural networks |
| `dcfr.py` | DeepCFR implementation and training loop |
| `encoding.py` | Infoset encoder for neural network input |
| `eval.py` | Evaluation script for trained policy network |

---

## Installation

```bash
git clone https://github.com/mahdiiii04/DeepCFR.git
cd DeepCFR
pip install -r requirements.txt
````

---

## Training

Train the Deep CFR agent:

```bash
python -c "from dcfr import DeepCFR; DeepCFR().train(iterations=2000)"
```

This will:

1. Sample cards for both players
2. Traverse the Kuhn Poker game tree
3. Store training examples in buffers
4. Train the regret and policy networks iteratively

---

## Evaluation

Use the policy network to play episodes and compute win rates:

```bash
python -c "from eval import play_episode; from dcfr import DeepCFR; agent = DeepCFR().policy_net; print(play_episode(agent))"
```

---

## Learn More

Read the detailed blog post:
[DeepCFR: When Regret Meets Neural Networks](https://mahdiiii04.github.io/posts/DeepCFR/)
