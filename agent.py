import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, env, state_size, n_action, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.9999, exploration_min=0.01, memory_size=10000, batch_size=64):
        self.env = env
        self.state_size = state_size
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory = []
        self.losses = []

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.n_action, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if state.shape != (self.state_size,):
            print("Adjusting state size from {} to {}".format(self.state_size, state.shape[0]))
            self.state_size = state.shape[0]
            self.model = self.build_model()
            self.target_model = self.build_model()
            self.target_model.set_weights(self.model.get_weights())
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.n_action)
        q_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])
    
    def update(self, state, action, reward, next_state, done):
        if state.shape != (self.state_size,):
            print("Adjusting state size from {} to {}".format(self.state_size, state.shape[0]))
            self.state_size = state.shape[0]
            self.model = self.build_model()
            self.target_model = self.build_model()
            self.target_model.set_weights(self.model.get_weights())
        target = reward
        if not done:
            target += self.discount_factor * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
    
        # Store the experience in memory
        self.memory.append((state, action, reward, next_state, done))  
    
        # Reduce memory size to memory_size
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]
    
        # Update the model
        target_f = self.model.predict(np.array(state).reshape(1, -1))
        target_f[0][action] = target
        history = self.model.fit(np.array(state).reshape(1, -1), target_f, epochs=1, verbose=0)
        self.losses.append(history.history['loss'][0])  # Record the loss
    
        # Decay the exploration rate
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        # Sample batch from the memory
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            self.update(state, action, reward, next_state, done)

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses)
        plt.title('Loss per update')
        plt.show()




















# import numpy as np
# import random
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt

# class DQNAgent:
#     def __init__(self, env, state_size, n_action, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.9999, exploration_min=0.01, memory_size=10000, batch_size=64):
#         self.env = env
#         self.state_size = state_size
#         self.n_action = n_action
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate
#         self.exploration_decay = exploration_decay
#         self.exploration_min = exploration_min
#         self.memory_size = memory_size
#         self.batch_size = batch_size
#         self.model = self.build_model()
#         self.target_model = self.build_model()
#         self.target_model.set_weights(self.model.get_weights())
#         self.memory = []
#         self.losses = []

#     def build_model(self):
#         model = Sequential()
#         model.add(Dense(32, input_shape=(self.state_size,), activation='relu'))
#         model.add(Dense(32, activation='relu'))
#         model.add(Dense(self.n_action, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
#         return model

#     def act(self, state):
#         print("State shape: ", state.shape)
#         print("Expected shape: ", self.state_size)
#         assert state.shape == (self.state_size,), "Invalid state shape"
#         if np.random.rand() < self.exploration_rate:
#             return random.randrange(self.n_action)
#         q_values = self.model.predict(state.reshape(1, -1))
#         return np.argmax(q_values[0])

#     def update(self, state, action, reward, next_state, done):
#         print("State shape: ", state.shape)
#         print("Expected shape: ", self.state_size)
#         assert state.shape == (self.state_size,), "Invalid state shape"
#         assert next_state.shape == (self.state_size,), "Invalid next_state shape"
#         target = reward
#         if not done:
#             target += self.discount_factor * np.amax(self.model.predict(next_state.reshape(1, -1))[0])

#         # Store the experience in memory
#         self.memory.append((state, action, reward, next_state, done))  

#         # Reduce memory size to memory_size
#         if len(self.memory) > self.memory_size:
#             self.memory = self.memory[-self.memory_size:]

#         # Update the model
#         target_f = self.model.predict(np.array(state).reshape(1, -1))
#         target_f[0][action] = target
#         history = self.model.fit(np.array(state).reshape(1, -1), target_f, epochs=1, verbose=0)
#         self.losses.append(history.history['loss'][0])  # Record the loss

#         # Decay the exploration rate
#         if self.exploration_rate > self.exploration_min:
#             self.exploration_rate *= self.exploration_decay


#     def replay(self):
#         if len(self.memory) < self.batch_size:
#             return
#         # Sample batch from the memory
#         batch = random.sample(self.memory, self.batch_size)
#         for state, action, reward, next_state, done in batch:
#             self.update(state, action, reward, next_state, done)

#     def update_target_model(self):
#         # Copy weights from model to target_model
#         self.target_model.set_weights(self.model.get_weights())

#     def plot_loss(self):
#         plt.figure(figsize=(10, 5))
#         plt.plot(self.losses)
#         plt.title('Loss per update')
#         plt.show()


















# import numpy as np
# import random
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam

# class DQNAgent:
#     def __init__(self, env, state_size, n_action, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.9999, exploration_min=0.01):
#         self.env = env
#         self.state_size = state_size
#         self.n_action = n_action
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate
#         self.exploration_decay = exploration_decay
#         self.exploration_min = exploration_min
#         self.model = self.build_model()
#         self.target_model = self.build_model()  # Initialize the target model
#         self.target_model.set_weights(self.model.get_weights())  # Set the target model weights to match the model's weights
#         self.epsilon = 1.0  # exploration rate
#         self.memory = []


#     def build_model(self):
#         model = Sequential()
#         model.add(Dense(32, input_shape=(76,), activation='relu'))  # Adjust input_shape
#         model.add(Dense(32, activation='relu'))
#         model.add(Dense(self.n_action, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  # Use 'learning_rate' argument
#         return model

#     def get_action(self, state):
#         if np.random.rand() < self.exploration_rate:
#             return random.randrange(self.n_action)
#         q_values = self.model.predict(state.reshape(1, -1))
#         return np.argmax(q_values[0])

#     def update(self, state, action, reward, next_state, done):
#         target = reward
#         if not done:
#             print("Next state shape:", next_state.shape)
#             next_state_reshaped = next_state.reshape(1, -1)
#             print("Next state reshaped shape:", next_state_reshaped.shape)
#             target += self.discount_factor * np.amax(self.model.predict(next_state_reshaped)[0])

#         # Store the experience in memory
#         self.memory.append((state, action, reward, next_state, done))  

#         # Update the model
#         target_f = self.model.predict(np.array(state).reshape(1, -1))
#         target_f[0][action] = target
#         self.model.fit(np.array(state).reshape(1, -1), target_f, epochs=1, verbose=0)

#         # Decay the exploration rate
#         if self.exploration_rate > self.exploration_min:
#             self.exploration_rate *= self.exploration_decay

#     def act(self, state):
#         # Exploration vs exploitation
#         if np.random.rand() <= self.epsilon:
#             # Take a random action
#             return random.randrange(self.n_action)
        
#         # Predict the reward value based on the given state
#         act_values = self.model.predict(state)

#         # Pick the action based on the predicted reward
#         return np.argmax(act_values[0])  # returns action

#     def replay(self, batch_size):
#         # Sample batch from the memory
#         batch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in batch:
#             target = reward
#             if not done:
#                 target += self.discount_factor * np.amax(self.model.predict(next_state.reshape(1, -1))[0])                
#             target_f = self.model.predict(state.reshape(1, -1))
#             target_f[0][action] = target
#             self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
#         if self.exploration_rate > self.exploration_min:
#             self.exploration_rate *= self.exploration_decay

#     def update_target_model(self):
#         # Copy weights from model to target_model
#         self.target_model.set_weights(self.model.get_weights())




















# import numpy as np
# import random
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam

# class DQNAgent:
#     def __init__(self, env, state_size, n_action, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.9999, exploration_min=0.01, memory_size=10000, batch_size=64):
#         self.env = env
#         self.state_size = state_size
#         self.n_action = n_action
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate
#         self.exploration_decay = exploration_decay
#         self.exploration_min = exploration_min
#         self.model = self.build_model()
#         self.target_model = self.build_model()  # Initialize the target model
#         self.target_model.set_weights(self.model.get_weights())  # Set the target model weights to match the model's weights
#         self.memory = []
#         self.memory_size = memory_size
#         self.batch_size = batch_size

#     def build_model(self):
#         model = Sequential()
#         model.add(Dense(32, input_shape=(76,), activation='relu'))  # Adjust input_shape
#         model.add(Dense(32, activation='relu'))
#         model.add(Dense(self.n_action, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  # Use 'learning_rate' argument
#         return model

#     def act(self, state): 
#         assert state.shape == (self.state_size,), "Invalid state shape"
#         if np.random.rand() < self.exploration_rate:
#             return random.randrange(self.n_action)
#         q_values = self.model.predict(state.reshape(1, -1))
#         return np.argmax(q_values[0])

#     def update(self, state, action, reward, next_state, done):
#         assert state.shape == (self.state_size,), "Invalid state shape"
#         assert next_state.shape == (self.state_size,), "Invalid next_state shape"
#         target = reward
#         if not done:
#             target += self.discount_factor * np.amax(self.model.predict(next_state.reshape(1, -1))[0])

#         # Store the experience in memory
#         self.memory.append((state, action, reward, next_state, done))  

#         # Reduce memory size to memory_size
#         if len(self.memory) > self.memory_size:
#             self.memory = self.memory[-self.memory_size:]

#         # Update the model
#         target_f = self.model.predict(np.array(state).reshape(1, -1))
#         target_f[0][action] = target
#         self.model.fit(np.array(state).reshape(1, -1), target_f, epochs=1, verbose=0)

#         # Decay the exploration rate
#         if self.exploration_rate > self.exploration_min:
#             self.exploration_rate *= self.exploration_decay

#     def replay(self):
#         if len(self.memory) < self.batch_size:
#             return
#         # Sample batch from the memory
#         batch = random.sample(self.memory, self.batch_size)
#         for state, action, reward, next_state, done in batch:
#             self.update(state, action, reward, next_state, done)

#     def update_target_model(self):
#         # Copy weights from model to target_model
#         self.target_model.set_weights(self.model.get_weights())























# import numpy as np
# import random
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam

# class DQNAgent:
#     def __init__(self, env, state_size, n_action, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.9999, exploration_min=0.01):
#         self.env = env
#         self.state_size = state_size
#         self.n_action = n_action
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate
#         self.exploration_decay = exploration_decay
#         self.exploration_min = exploration_min
#         self.model = self.build_model()
#         self.target_model = self.build_model()  # Initialize the target model
#         self.target_model.set_weights(self.model.get_weights())  # Set the target model weights to match the model's weights
#         self.epsilon = 1.0  # exploration rate
#         self.memory = []


#     def build_model(self):
#         model = Sequential()
#         model.add(Dense(32, input_shape=(76,), activation='relu'))  # Adjust input_shape
#         model.add(Dense(32, activation='relu'))
#         model.add(Dense(self.n_action, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  # Use 'learning_rate' argument
#         return model

#     def get_action(self, state):
#         if np.random.rand() < self.exploration_rate:
#             return random.randrange(self.n_action)
#         q_values = self.model.predict(state.reshape(1, -1))
#         return np.argmax(q_values[0])

#     def update(self, state, action, reward, next_state, done):
#         target = reward
#         if not done:
#             print("Next state shape:", next_state.shape)
#             next_state_reshaped = next_state.reshape(1, -1)
#             print("Next state reshaped shape:", next_state_reshaped.shape)
#             target += self.discount_factor * np.amax(self.model.predict(next_state_reshaped)[0])

#         # Store the experience in memory
#         self.memory.append((state, action, reward, next_state, done))  

#         # Update the model
#         target_f = self.model.predict(np.array(state).reshape(1, -1))
#         target_f[0][action] = target
#         self.model.fit(np.array(state).reshape(1, -1), target_f, epochs=1, verbose=0)

#         # Decay the exploration rate
#         if self.exploration_rate > self.exploration_min:
#             self.exploration_rate *= self.exploration_decay

#     def act(self, state):
#         # Exploration vs exploitation
#         if np.random.rand() <= self.epsilon:
#             # Take a random action
#             return random.randrange(self.n_action)
        
#         # Predict the reward value based on the given state
#         act_values = self.model.predict(state)

#         # Pick the action based on the predicted reward
#         return np.argmax(act_values[0])  # returns action

#     def replay(self, batch_size):
#         # Sample batch from the memory
#         batch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in batch:
#             target = reward
#             if not done:
#                 target += self.discount_factor * np.amax(self.model.predict(next_state.to_numpy().reshape(1, -1))[0])
#             target_f = self.model.predict(state.reshape(1, -1))
#             target_f[0][action] = target
#             self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
#         if self.exploration_rate > self.exploration_min:
#             self.exploration_rate *= self.exploration_decay

#     def update_target_model(self):
#         # Copy weights from model to target_model
#         self.target_model.set_weights(self.model.get_weights())










# import numpy as np
# import os
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam

# class DQNAgent:
#     def __init__(self, n_action, state_size=50, alpha=0.5, gamma=0.95, epsilon=0.1, model_path=None):
#         self.state_size = state_size
#         self.n_action = n_action
#         self.memory = []
#         self.gamma = gamma  # discount rate
#         self.epsilon = epsilon  # exploration rate
#         self.alpha = alpha  # learning rate

#         if model_path is not None:
#             if os.path.exists(model_path):
#                 self.model = load_model(model_path)  # Loading a pre-trained CNN/hybrid model
#             else:
#                 raise Exception(f"No model found at {model_path}")
#         else:
#             # Handle the case where no model path is provided
#             print("No model path provided. Initializing a new model.")
#             self.model = self._create_model()

#     def _create_model(self):
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.n_action, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
#         return model

#     def get_action(self, state):
#         assert state.shape == (self.state_size,), "Input state shape is incorrect."

#         # epsilon-greedy policy
#         if np.random.rand() <= self.epsilon:
#             action = np.random.choice(self.n_action)  # explore
#         else:
#             q_values = self.model.predict(state[np.newaxis, :])
#             action = np.argmax(q_values[0])  # exploit

#         return action

#     def update(self, state, action, reward, next_state, done):
#         assert state.shape == (self.state_size,), "Input state shape is incorrect."

#         target = self.model.predict(state[np.newaxis, :])
#         if done:
#             target[0][action] = reward
#         else:
#             q_future = max(self.model.predict(next_state[np.newaxis, :])[0])
#             target[0][action] = reward + q_future * self.gamma
#         self.model.fit(state[np.newaxis, :], target, epochs=1, verbose=0)











# import numpy as np
# import os
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam

# class DQNAgent:
#     def __init__(self, n_action, state_size=50, alpha=0.5, gamma=0.95, epsilon=0.1, model_path=None):
#         self.state_size = state_size
#         self.n_action = n_action
#         self.memory = []
#         self.gamma = gamma  # discount rate
#         self.epsilon = epsilon  # exploration rate
#         self.alpha = alpha  # learning rate
#         self.check_input_shapes = True

#         if model_path is not None:
#             if os.path.exists(model_path):
#                 self.model = load_model(model_path)  # Loading a pre-trained CNN/hybrid model
#             else:
#                 raise Exception(f"No model found at {model_path}")
#         else:
#             # Handle the case where no model path is provided
#             print("No model path provided. Please provide a path to a pre-trained model.")
#             self.model = self._create_model()

#     def _create_model(self):
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.n_action, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
#         return model

#     def get_action(self, state):
#         if self.check_input_shapes:
#             print(f"Agent input shape in get_action: {state.shape}")

#         # epsilon-greedy policy
#         if np.random.rand() <= self.epsilon:
#             action = np.random.choice(self.n_action)  # explore
#         else:
#             q_values = self.model.predict(state[np.newaxis, :])
#             action = np.argmax(q_values[0])  # exploit

#         if self.check_input_shapes:
#             print(f"Agent output (selected action) in get_action: {action}")
#             self.check_input_shapes = False

#         return action

#     def update(self, state, action, reward, next_state, done):
#         if self.check_input_shapes:
#             print(f"Agent input shape in update: {state.shape}")

#         target = self.model.predict(state)
#         if done:
#             target[0][action] = reward
#         else:
#             q_future = max(self.model.predict(next_state)[0])
#             target[0][action] = reward + q_future * self.gamma
#         self.model.fit(state, target, epochs=1, verbose=0)















# import numpy as np
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam

# class DQNAgent:
#     def __init__(self, n_action, state_size=76, alpha=0.5, gamma=0.95, epsilon=0.1, model_path=None):
#         self.state_size = state_size
#         self.n_action = n_action
#         self.memory = []
#         self.gamma = gamma  # discount rate
#         self.epsilon = epsilon  # exploration rate
#         self.alpha = alpha  # learning rate

#         if model_path is not None:
#             self.model = load_model('C:/Users/janni/OneDrive/Skrivebord/VS Code/hybrid_model.h5')  # Loading a pre-trained CNN/hybrid model
#         else:
#             # Handle the case where no model path is provided
#             print("No model path provided. Please provide a path to a pre-trained model.")

#     def _create_model(self):
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.n_action, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
#         return model

#     def get_action(self, state):
#         # epsilon-greedy policy
#         if np.random.rand() <= self.epsilon:
#             return np.random.choice(self.n_action)  # explore
#         else:
#             print(state.shape)
#             q_values = self.model.predict(state[np.newaxis, :])

#             return np.argmax(q_values[0])  # exploit

#     def update(self, state, action, reward, next_state, done):
#         target = self.model.predict(state)
#         if done:
#             target[0][action] = reward
#         else:
#             q_future = max(self.model.predict(next_state)[0])
#             target[0][action] = reward + q_future * self.gamma
#         self.model.fit(state, target, epochs=1, verbose=0)







# import numpy as np
# from tensorflow.keras.models import load_model

# class QLearningAgent:
#     def __init__(self, n_action, alpha=0.5, gamma=0.95, epsilon=0.1, model_path=None):
#         self.n_action = n_action
#         self.alpha = alpha  # learning rate
#         self.gamma = gamma  # discount factor
#         self.epsilon = epsilon  # exploration rate

#         # Load the hybrid model if a path is provided
#         if model_path is not None:
#             self.hybrid_model = load_model(model_path)
#         else:
#             self.hybrid_model = None

#         # Initialize Q-table with zero
#         self.Q = {} 

#     def get_state(self, state):
#         # If the hybrid model is loaded, use it to generate features for the state
#         if self.hybrid_model is not None:
#             state_features = self.hybrid_model.predict(state.reshape(1, -1)).flatten()
#             return str(state_features)
#         else:
#             return str(state)

#     def get_action(self, state):
#         # epsilon-greedy policy
#         if np.random.random() < self.epsilon:
#             return np.random.choice(self.n_action)  # explore
#         else:
#             state = self.get_state(state)
#             if state not in self.Q:
#                 return np.random.choice(self.n_action) 
#             return np.argmax(self.Q[state])  # exploit

#     def update(self, state, action, reward, next_state):
#         state = self.get_state(state)
#         next_state = self.get_state(next_state)
#         if state not in self.Q:
#             self.Q[state] = np.zeros(self.n_action)
#         if next_state not in self.Q:
#             self.Q[next_state] = np.zeros(self.n_action)
        
#         # Q-Learning update
#         self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])




# import numpy as np
# from tensorflow.keras.models import load_model

# class QLearningAgent:
#     def __init__(self, n_action, n_bins, alpha=0.5, gamma=0.95, epsilon=0.1, model_path=None):
#         self.n_action = n_action
#         self.n_bins = n_bins
#         self.Q = np.zeros((n_bins,) * 5 + (n_action,))  # initialize Q-table with new dimension
#         self.alpha = alpha  # learning rate
#         self.gamma = gamma  # discount factor
#         self.epsilon = epsilon  # exploration rate

#         # Load the hybrid model if a path is provided
#         if model_path is not None:
#             self.hybrid_model = load_model(model_path)
#         else:
#             self.hybrid_model = None

#     def discretize_state_features(self, state_features):
#         bins = np.linspace(-1, 1, self.n_bins)  # Adjust the range and number of bins based on your problem
#         discretized_state = np.digitize(state_features, bins)
#         return tuple(discretized_state - 1)  # Subtract 1 to make the bin indices start from 0

#     def get_action(self, state):
#         # epsilon-greedy policy
#         if np.random.random() < self.epsilon:
#             return np.random.choice(self.n_action)  # explore
#         else:
#             # If the hybrid model is loaded, use it to generate features for the state
#             if self.hybrid_model is not None:
#                 state_features = self.hybrid_model.predict(state.reshape(1, -1)).flatten()
#                 state = self.discretize_state_features(state_features)
#             return np.argmax(self.Q[state])  # exploit

#     def update(self, state, action, reward, next_state):
#         # Q-Learning update
#         self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
