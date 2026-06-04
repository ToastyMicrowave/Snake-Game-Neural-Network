# Snake — Deep Reinforcement Learning from Scratch

A self-playing Snake agent trained with **Deep Q-Learning**, where *every* learning
component — the neural network, the convolutional layers, backpropagation, and the
DQN training loop — is implemented **from scratch in pure NumPy**. No PyTorch, no
TensorFlow, no autograd. Just linear algebra and calculus written by hand.

The goal of this project was to understand deep RL by building it, rather than calling
`model.fit()`. Every gradient in this repo was derived and coded manually.

---

## Why this is interesting

Most "Snake AI" projects import a framework that hides the math. This one doesn't.
The repository contains a complete, working implementation of:

- **A feedforward neural network** with manual forward and backward passes
- **A convolutional neural network** (2D convolution + ReLU + max-pooling) with a
  full hand-derived backward pass, including gradient routing through max-pool
  "switches"
- **Double DQN** reinforcement learning with a target network, experience replay,
  and epsilon-greedy exploration
- **A from-scratch Snake environment** exposing a clean RL interface
  (`reset` / `step` / `render`)

If you want to see whether someone actually understands backpropagation and
Q-learning — not just the API surface — this is that code.

---

## Architecture

```
                 ┌─────────────────────────────────────────┐
                 │            Snake environment             │
                 │   reset() · step(action) · render()      │
                 └───────────────┬─────────────────────────-┘
                                 │ observation (state)
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │                  Double DQN loop                    │
        │                                                     │
        │   ε-greedy action ──► env.step ──► ReplayBuffer     │
        │            ▲                              │         │
        │            │       sample minibatch       ▼         │
        │      online network  ◄── TD target ── target network│
        │       (trains)            (frozen, periodic sync)   │
        └────────────────────────────────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │         NeuralNetwork  (pure NumPy)                 │
        │                                                     │
        │   Dense path:   X → W1 → ReLU → W2 → Q-values       │
        │   Conv path:    grid → Conv2D → ReLU → MaxPool      │
        │                      → flatten → Dense → Q-values   │
        └────────────────────────────────────────────────────┘
```

### Files

| File | Contents |
|------|----------|
| [neural_network.py](neural_network.py) | `NeuralNetwork` (dense + conv) and `ReplayBuffer` |
| [snake.py](snake.py) | `Snake` environment + the Double DQN training loop |

---

## The neural network ([neural_network.py](neural_network.py))

### Dense network
A two-layer MLP: `input → ReLU(hidden) → linear(output)`. The linear output head is
what makes it a Q-function approximator (Q-values are unbounded, so no squashing
activation on the output).

- **Forward pass** — explicit matrix algebra (`forward`)
- **Backward pass** — the chain rule written out by hand (`backward`), computing
  `dW1, db1, dW2, db2` from the output error
- **Gradient clipping** — gradients are clipped element-wise to `[-1, 1]` before the
  SGD update to stabilize the notoriously unstable bootstrapped DQN targets
- **Weight management** — `get_weights` / `set_weights` exist specifically to support
  the target-network copy in DQN

### Convolutional network
A complete CNN forward *and* backward pass, written without any framework:

- **`convolve2d`** — 2D cross-correlation with configurable padding and stride
- **`max_pool`** — max pooling that also records **switch masks** (which input cell won
  each pooling window) so gradients can be routed back correctly
- **`extract_features`** — the full conv forward pass (pad → convolve → ReLU → pool →
  flatten), caching every intermediate needed for backprop
- **`conv_backward`** — the hard part: it un-pools the gradient through the saved
  switches, applies the ReLU derivative, and accumulates the **kernel gradient**
  `dk` by correlating the upstream gradient against the cached input patches
- **`conv_train_batch`** — stitches the conv path and the dense path into one
  end-to-end gradient step

This means the agent can, in principle, learn directly from the **raw board grid**
instead of hand-engineered features — the convolutional machinery to do so is fully
present and differentiable.

### Experience replay
`ReplayBuffer` is a fixed-capacity `deque` of transitions
`(state, action, reward, next_state, done)` with uniform random sampling, returned as
batched NumPy arrays ready to feed the network.

---

## The environment ([snake.py](snake.py))

A dependency-light Snake game built to a standard RL contract.

- **`reset()`** → returns the initial observation
- **`step(action)`** → returns `(observation, reward, done)`
- **`render()`** → optional pygame visualization

### State representation
The observation is a 13-dimensional feature vector designed to be both compact and
informative:

- **Danger sensors** — collision risk straight ahead, to the relative left, and to the
  relative right (computed in the snake's own frame of reference, not absolute
  directions)
- **Food direction** — boolean up/down/left/right flags
- **Current heading** — one-hot of the four movement directions
- **Normalized food offset** — `(dx, dy)` scaled by grid size

### Reward shaping
- Large positive reward for eating food
- Small **distance-based shaping** term (`old_dist - new_dist`) that nudges the snake
  toward the food every step, giving dense feedback instead of a sparse "eat or
  nothing" signal
- Penalties for dying and for stalling (a `max_steps` timeout prevents the snake from
  looping forever without eating)

---

## The training loop ([snake.py](snake.py))

A textbook **Double DQN** setup:

1. **ε-greedy exploration** — start fully random (`ε = 1`) and decay toward a small
   floor, shifting from exploration to exploitation as the agent improves
2. **Experience replay** — store every transition; train on random minibatches to
   break the temporal correlation that would otherwise destabilize learning
3. **Double DQN target** — the **online** network *chooses* the best next action, but
   the **target** network *evaluates* it. This decoupling is the Double-DQN fix for
   the overestimation bias of vanilla DQN:

   ```
   a*      = argmax_a  Q_online(s', a)
   target  = r + γ · Q_target(s', a*) · (1 - done)
   ```

4. **Target network sync** — the target network's weights are periodically hard-copied
   from the online network, giving stable bootstrap targets between syncs

---

## Running it

```bash
pip install numpy pygame
python snake.py
```

This launches training and then a visualized evaluation. Tunable hyperparameters
(grid size, hidden width, ε schedule, γ, batch size, replay capacity, sync interval)
live at the top of the `__main__` block in [snake.py](snake.py).

---

## Concepts demonstrated

- Manual backpropagation through dense **and** convolutional layers
- Max-pool gradient routing via switch masks
- The Bellman equation and temporal-difference learning
- Double DQN, target networks, and experience replay
- Reward shaping and state-feature engineering for RL
- A clean, framework-agnostic environment API

---

*Built to learn how deep RL actually works underneath the frameworks — by deriving and
implementing every piece by hand.*
