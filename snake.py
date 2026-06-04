import pygame, random, neural_network
import numpy as np
import time
from collections import deque

ACTIONS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1)    # RIGHT
}

# ─── Hyperparameters ────────────────────────────────────────────────
# Central place to tune the agent. Grouped by concern so experiments
# only ever touch this dict, never the training loop below.
CONFIG = {
    # Environment / network shape
    "grid_size":         25,        # board is grid_size x grid_size cells
    "hidden_size":       128,      # width of the hidden layer
    "lr":                0.003,     # SGD learning rate for the networks
    "grad_clip_norm":    5.0,       # clip the global gradient L2 norm to this

    # Exploration (epsilon-greedy)
    "epsilon_start":      1.0,      # fully random at the start
    "epsilon_min":        0.005,    # exploration floor
    "epsilon_decay_frac": 0.6,      # reach epsilon_min after this fraction of train_steps
                                    # (per-step decay rate is derived from train_steps below)

    # Experience replay / batching
    "replay_capacity":   100_000,   # max transitions held in the buffer
    "batch_size":        100,       # minibatch size per gradient step
    "train_every":       2,         # take a gradient step every N env steps

    # Q-learning
    "gamma":             0.99,      # discount factor for future reward
    "target_sync_every": 1000,       # hard-copy online -> target weights every N steps

    # Rewards (env returns these from step())
    "reward_food":          20,     # base reward for eating food
    "reward_game_over":     -15,    # collision with a wall or itself
    "reward_timeout":       -15,    # too many steps without eating
    "reward_move":          -0.001,   # per-step living penalty -> rewards efficiency
    "reward_shaping_scale":  1,   # reward per cell of progress toward food (grid-independent)

    # Run length
    "train_steps":       100_000,    # total environment steps to train for
    "eval_episodes":     5000,      # episodes to run after training
    "render_eval":       True,      # visualize the evaluation games
    "progress_every":    1000,      # refresh the training progress line every N steps
}


def format_eta(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"

class Snake:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size

        self.reset()
    
    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = random.choice(tuple(ACTIONS.values()))
        self.spawn_food()
        self.score = 0
        self.game_over = False
        self.max_steps = self.grid_size * 10
        self.steps_since_food = 0

        
        return self.get_observation()
    
    def spawn_food(self):
        empty_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in self.snake]
        self.food = random.choice(empty_cells)
        
    def get_observation(self):
        head_x, head_y = self.snake[0]

        dir_l = self.direction == (0, -1)
        dir_r = self.direction == (0, 1)
        dir_u = self.direction == (-1, 0)
        dir_d = self.direction == (1, 0)

        def danger(p):
            return (
                p[0] < 0 or p[0] >= self.grid_size or
                p[1] < 0 or p[1] >= self.grid_size or
                p in self.snake
            )

        # Relative left/right/straight
        danger_straight = danger((
            head_x + self.direction[0],
            head_y + self.direction[1]
        ))
        danger_right = danger((
            head_x + self.direction[1],
            head_y - self.direction[0]
        ))
        danger_left = danger((
            head_x - self.direction[1],
            head_y + self.direction[0]
        ))

        food_x, food_y = self.food
        
        dx = (food_x - head_x) / self.grid_size
        dy = (food_y - head_y) / self.grid_size


        food_up = food_y < head_y
        food_down = food_y > head_y
        food_left = food_x < head_x
        food_right = food_x > head_x

        return np.array([
            danger_straight,
            danger_right,
            danger_left,
            food_up,
            food_down,
            food_left,
            food_right,
            dir_u,
            dir_d,
            dir_l,
            dir_r,
            dx,
            dy
        ])


    def is_opposite(self, dir1, dir2):
        return dir1[0] == -dir2[0] and dir1[1] == -dir2[1]

    def step(self, action):
        self.steps_since_food += 1
        if self.steps_since_food >= self.max_steps or self.game_over:
            return self.get_observation(), CONFIG["reward_timeout"], True
        
        hx, hy = self.snake[0]
        fx, fy = self.food
        # Raw cell distance (NOT normalized by grid_size) so the shaping signal
        # is just as strong on a 30x30 board as on a 10x10 one. A move that closes
        # the gap by one cell is worth reward_shaping_scale regardless of grid size.
        old_dist = np.linalg.norm([hx - fx, hy - fy])
        
        new_direction = ACTIONS[action]
        if not self.is_opposite(new_direction, self.direction) or len(self.snake) == 1:
            self.direction = new_direction
        
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            self.game_over = True
            return self.get_observation(), CONFIG["reward_game_over"], True
        
        
        self.snake.insert(0, new_head)
        new_dist = np.linalg.norm([new_head[0] - fx, new_head[1] - fy])
        delta = old_dist - new_dist

        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            self.spawn_food()
            # Big reward for eating food, plus shaped bonus for closing distance
            reward = CONFIG["reward_food"] + (delta * CONFIG["reward_shaping_scale"])
        else:
            self.snake.pop()
            # Per-step penalty, plus shaped reward for getting closer / penalty for moving away
            reward = CONFIG["reward_move"] + (delta * CONFIG["reward_shaping_scale"])

        return self.get_observation(), reward, self.game_over
    
    def render(self, scale=20):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * scale, self.grid_size * scale))
            self.clock = pygame.time.Clock()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((0, 0, 0))
        fx, fy = self.food
        pygame.draw.rect(self.screen, (0, 255, 0), (fy * scale, fx * scale, scale, scale))
        
        for x, y in self.snake:
            pygame.draw.rect(self.screen, (0, 0, 255), (y * scale, x * scale, scale, scale))
        
        pygame.display.flip()
        self.clock.tick(30)
        
        

def test_agent(env, model, episodes=5, render=True):
    total_score = 0

    for ep in range(episodes):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            if render:
                env.render()

            q_values = model.predict(obs)
            action = np.argmax(q_values)
            obs, reward, done = env.step(action)

        print(f"Test Episode {ep+1} | Score: {env.score}")
        total_score += env.score

    avg_score = total_score / episodes
    print(f"\nAverage Score over {episodes} test episodes: {avg_score}")
    return avg_score


        
if __name__ == "__main__":
    cfg = CONFIG

    env = Snake(cfg["grid_size"])
    observation = env.reset()
    done = False
    inputs = len(observation)

    nn = neural_network.NeuralNetwork(
        input_size=inputs, hidden_size=cfg["hidden_size"], output_size=len(ACTIONS),
        lr=cfg["lr"], grad_clip=cfg["grad_clip_norm"]
    )
    target_nn = neural_network.NeuralNetwork(
        input_size=inputs, hidden_size=cfg["hidden_size"], output_size=len(ACTIONS),
        lr=cfg["lr"], grad_clip=cfg["grad_clip_norm"]
    )

    epsilon = cfg["epsilon_start"]
    batch_size = cfg["batch_size"]
    gamma = cfg["gamma"]
    replay_buffer = neural_network.ReplayBuffer(max_size=cfg["replay_capacity"])

    # Derive the per-step decay so epsilon reaches its floor after
    # epsilon_decay_frac of training, no matter how long train_steps is.
    decay_steps = max(1, int(cfg["train_steps"] * cfg["epsilon_decay_frac"]))
    epsilon_decay = (cfg["epsilon_min"] / cfg["epsilon_start"]) ** (1 / decay_steps)

    epoch = 0
    # Progress tracking
    start_time = time.time()
    games_played = 0
    recent_scores = deque(maxlen=100)   # rolling window of finished-game scores
    # Training
    for i in range(cfg["train_steps"]):
        if random.random() < epsilon:
            action = random.randint(0, len(ACTIONS) - 1)
        else:
            q_values = nn.predict(observation)
            action = np.argmax(q_values)

        next_observation, reward, done = env.step(action)
        replay_buffer.store(observation, action, reward, next_observation, done)
        observation = next_observation

        if len(replay_buffer) >= batch_size and i % cfg["train_every"] == 0:
            observations, actions, rewards, next_observations, dones = replay_buffer.sample(batch_size)
            target_qs_online = nn.predict(observations)
            next_qs_online = nn.predict(next_observations)
            best_next_actions = np.argmax(next_qs_online, axis=1)

            next_qs_target = target_nn.predict(next_observations)
            max_next_qs = next_qs_target[np.arange(batch_size), best_next_actions]

            target_qs = target_qs_online.copy()
            targets = rewards + gamma * max_next_qs * (1 - dones)



            target_qs[np.arange(batch_size), actions] = targets

            nn.train_batch(observations, target_qs)
        epoch += 1
        if epoch % cfg["target_sync_every"] == 0:
            target_nn.set_weights(nn.get_weights())

        if done:
            recent_scores.append(env.score)
            games_played += 1
            observation = env.reset()
            done = False
        epsilon = max(epsilon * epsilon_decay, cfg["epsilon_min"])

        # Live progress line (updates in place, so it's cheap and not spammy)
        if (i + 1) % cfg["progress_every"] == 0:
            steps_done = i + 1
            elapsed = time.time() - start_time
            rate = steps_done / elapsed if elapsed > 0 else 0.0
            eta = (cfg["train_steps"] - steps_done) / rate if rate > 0 else 0.0
            pct = 100 * steps_done / cfg["train_steps"]
            avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
            print(
                f"\r[{pct:5.1f}%] step {steps_done:>7}/{cfg['train_steps']}  "
                f"ε={epsilon:.3f}  games={games_played:>5}  "
                f"avg_score={avg:4.1f}  {rate:5.0f} st/s  ETA {format_eta(eta)}   ",
                end="", flush=True,
            )

    print()  # finish the progress line before evaluation output
    test_agent(env, nn, episodes=cfg["eval_episodes"], render=cfg["render_eval"])


