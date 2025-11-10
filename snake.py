import pygame, random, neural_network
import numpy as np

ACTIONS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1)    # RIGHT
}

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

        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

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
            return self.get_observation(), -2, True
        
        hx, hy = self.snake[0]
        fx, fy = self.food
        old_dist = np.linalg.norm([hx - fx, hy - fy]) / self.grid_size
        
        new_direction = ACTIONS[action]
        if not self.is_opposite(new_direction, self.direction) or len(self.snake) == 1:
            self.direction = new_direction
        
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            self.game_over = True
            return self.get_observation(), -2, True
        
        
        self.snake.insert(0, new_head)
        new_dist = np.linalg.norm([new_head[0] - fx, new_head[1] - fy]) / self.grid_size
        delta = old_dist - new_dist

        if new_head == self.food:
            self.score += 1 
            print(self.score)
            self.steps_since_food = 0 
            self.spawn_food()
            reward = 1000
        else:
            self.snake.pop()
            reward = -0.004 + delta

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
        
        
        
        
if __name__ == "__main__":
    env = Snake(15)
    observation = env.reset()
    done = False
    inputs = len(observation)
    
    nn = neural_network.NeuralNetwork(input_size=inputs, hidden_size=1000, output_size=len(ACTIONS))
    target_nn = neural_network.NeuralNetwork(input_size=inputs, hidden_size=1000, output_size=len(ACTIONS))
    epsilon = 1
    epsilon_decay = 0.999
    epsilon_min = 0.005
    replay_buffer = neural_network.ReplayBuffer(max_size=100000)
    batch_size = 100
    gamma = 0.99
    epoch = 0 
    # Training
    for i in range(50000):
        
        if random.random() < epsilon:
            action = random.randint(0, len(ACTIONS) - 1)
        else:
            q_values = nn.predict(observation)
            action = np.argmax(q_values)
        
        next_observation, reward, done = env.step(action)
        replay_buffer.store(observation, action, reward, next_observation, done)
        observation = next_observation

        print(epoch, epsilon)
        if len(replay_buffer) >= batch_size and i % 2 == 0:
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
        if epoch % 500 == 0:
            target_nn.set_weights(nn.get_weights())
        
        if done:
            observation = env.reset()
            done = False
        epsilon = max(epsilon * epsilon_decay, epsilon_min)


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
        total_score += score

    avg_score = total_score / episodes
    print(f"\nAverage Score over {episodes} test episodes: {avg_score}")
    return avg_score

test_agent(env, nn, episodes=5000, render=True)