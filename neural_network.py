import numpy as np
from collections import deque
from random import sample as rsample
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, kernel=None):
        # Weights
        
        
        self.w1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        
        self.w2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        self.lr = lr

        if kernel is None:
            self.kernel = np.random.randn(3, 3) * 0.01
            
        else:
            self.kernel = kernel.astype(float)
        self.dk = np.zeros_like(self.kernel)
            
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(s):
        return s * (1 - s)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.w1.T + self.b1
        a1 = self.relu(self.z1)
        
        self.z2 = a1 @ self.w2.T + self.b2
        a2 = self.z2
        
        return a1, a2
    
    def backward(self, X ,y_true, a1, a2):
        delta_output = a2 - y_true
        self.dw2 = delta_output.T @ a1
        self.db2 = np.sum(delta_output, axis=0, keepdims=True)
        
        self.delta_hidden = (delta_output @ self.w2) * self.relu_derivative(a1)
        
        self.dw1 = self.delta_hidden.T @ X
        self.db1 = np.sum(self.delta_hidden, axis=0, keepdims=True)
        
    def update_weights(self):
        
        self.dw1 = np.clip(self.dw1, -1, 1)
        self.dw2 = np.clip(self.dw2, -1, 1)
        self.db1 = np.clip(self.db1, -1, 1)
        self.db2 = np.clip(self.db2, -1, 1)

        self.w1 -= self.lr * self.dw1
        self.b1 -= self.lr * self.db1

        self.w2 -= self.lr * self.dw2
        self.b2 -= self.lr * self.db2
        
        if self.dk is not None:
            self.kernel -= self.lr * self.dk
    
    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            a1, a2 = self.forward(X)
            self.backward(X, y, a1, a2)
            self.update_weights()
            
    def train_one(self, state, target_q):
        state = state.reshape(1, -1)
        target_q = target_q.reshape(1, -1)
        
        a1, a2 = self.forward(state)
        self.backward(state, target_q, a1, a2)
        self.update_weights()
        
    def train_batch(self, states, target_qs):
        a1s, a2s = self.forward(states)
        self.backward(states, target_qs, a1s, a2s)
        self.update_weights()

    def predict(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)  # single sample
            a1, a2 = self.forward(state)
            return a2.flatten()           
        else:
            # Already a batch
            a1, a2 = self.forward(state)
            return a2                     

    def extract_features(self, input_grids, pool_size=2, stride=2):
        arr = np.array(input_grids)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        feats = []
        self._conv_inputs = []
        self._conv_z = []
        self._pool_switches = []
        
        for g in arr:
            padded = np.pad(g, 1, mode='constant')
            self._conv_inputs.append(padded)
            z = self.convolve2d(padded, self.kernel)
            self._conv_z.append(z)
            
            r = self.relu(z)
            
            p, s = self.max_pool(r, size=pool_size, stride=stride)
            self._pool_switches.append(s)
            feats.append(p.flatten())
        return np.vstack(feats)
    def conv_backward(self, dX, pool_size=2, stride=2):
        batch = len(self._conv_inputs)
        conv_h, conv_w = self._conv_z[0].shape
        out_h = (conv_h - pool_size) // stride + 1
        out_w = (conv_w - pool_size) // stride + 1
        
        dZ = np.zeros((batch, conv_h, conv_w))
        self.dk = np.zeros_like(self.kernel)
        
        for b in range(batch):
            dpool = dX[b].reshape(out_h, out_w)
            switches = self._pool_switches[b]
            
            dr = np.zeros_like(self._conv_z[b])
            for i in range(out_h):
                for j in range(out_w):
                        dr[i*stride : i*stride+pool_size, j*stride : j*stride+pool_size] += switches[i, j] * dpool[i, j]
            
            dZ[b]  = dr * self.relu_derivative(self._conv_z[b])
            inp = self._conv_inputs[b]  
            gh, gw = self.kernel.shape
            for i in range(gh):
                for j in range(gw):
                    patch = inp[i : i + conv_h, j : j + conv_w]
                    self.dk[i, j] += np.sum(patch * dZ[b])
            
        self.dk /= batch
    def conv_train_batch(self, states, target_qs, pool_size=2, stride=2):
        X = self.extract_features(states, pool_size, stride)
        a1s, a2s = self.forward(X)                                 
        self.backward(X, target_qs, a1s, a2s)
        dX = self.delta_hidden @ self.w1
        self.conv_backward(dX, pool_size, stride)
        self.update_weights()
        
    def get_weights(self):
        return self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()

    def set_weights(self, weights):
        self.w1, self.b1, self.w2, self.b2 = [w.copy() for w in weights]
    
    def convolve2d(self, input, kernel, padding=0, stride=1):
        input = np.pad(np.array(input), [(padding, padding), (padding, padding)], constant_values=0)
        kernel = np.array(kernel)
        
        kernel_size = len(kernel)
        size = ((len(input) - kernel_size ) // stride) + 1
        output = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                patch = input[i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
                output[i, j] = np.sum(patch * kernel)
                
        return output
                
    def max_pool(self, input, size=2, stride=2):
        h, w = input.shape
        out_h = (h - size) // stride + 1
        out_w = (w - size) // stride + 1

        pooled   = np.zeros((out_h, out_w))
        switches = np.zeros((out_h, out_w, size, size), dtype=float)

        for i in range(out_h):
            for j in range(out_w):
                patch = input[i*stride:i*stride+size, j*stride:j*stride+size]
                idx   = np.unravel_index(np.argmax(patch), patch.shape)
                pooled[i,j]               = patch[idx]
                switches[i,j, idx[0], idx[1]] = 1.0

        return pooled, switches
    
    def conv_predict(self, input_grid, pool_size=2, stride=2):
        
        if isinstance(input_grid, np.ndarray) and input_grid.ndim == 3:
            return np.vstack([self.conv_predict(g, pool_size, stride) for g in input_grid])
            
        # Convolve
        conv_output = self.convolve2d(input_grid, self.kernel, padding=1)
        
        activated = self.relu(conv_output)
        
        # MaxPool
        pooled, _ = self.max_pool(activated, size=pool_size, stride=stride)
        
        return self.predict(pooled.flatten())

    
class ReplayBuffer:
    def __init__(self, max_size = 50):
        self.memory = deque([], max_size)

    def __len__(self):
        return len(self.memory)
    
    def store(self, state, action, reward, next_state, done):
        self.memory.append({'s' : state, 'a': action, 'r': reward, 'n_s': next_state, 'd': done})
        
    def sample(self, size): 
        batch = rsample(self.memory, min(size, len(self.memory)))
        states = np.array([sample['s'] for sample in batch])
        actions = np.array([sample['a'] for sample in batch])
        rewards = np.array([sample['r'] for sample in batch])
        next_states = np.array([sample['n_s'] for sample in batch])
        dones = np.array([sample['d'] for sample in batch])
        return states, actions, rewards, next_states, dones
    
