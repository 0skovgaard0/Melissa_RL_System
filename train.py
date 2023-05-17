import os

class Trainer:

    def __init__(self, env, agent, initial_prices, price_updates):
        self.env = env
        self.agent = agent
        self.initial_prices = initial_prices
        self.price_updates = price_updates

        # Define the directory for saving the model
        self.model_dir = 'models'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.rewards = []
        self.net_worths = []
        self.best_reward = -float('inf')
        self.no_improvement_threshold = 50
        self.episodes_without_improvement = 0

    def train(self):
        n_episode = 1000
        for episode in range(n_episode):
            self.env.reset(self.initial_prices)
            total_reward = 0
            i = 0
            while True:
                if i == len(self.price_updates):
                    break
                state = self.env.get_state()
                print(f"Shape of data passed to get_state: Initial prices - {self.initial_prices.shape}, Price updates - {self.price_updates.iloc[i].shape}")  # Debug print statement
                action = self.agent.act(state)  
                next_state, reward, trades, done = self.env.step(action, self.price_updates.iloc[i])
                self.agent.update(state, action, reward, next_state, done)
                total_reward += reward
                i += 1
                if done:
                    break


            self.rewards.append(total_reward)
            self.net_worths.append(self.env.get_net_worth())
            print(f'Episode: {episode}, Total Reward: {total_reward}, Net Worth: {self.env.get_net_worth()}')

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1

            if self.episodes_without_improvement >= self.no_improvement_threshold:
                print(f'No improvement in {self.no_improvement_threshold} episodes, stopping training...')
                break

            # Save the model weights every 100 episodes
            if episode % 100 == 0:
                self.agent.model.save_weights(os.path.join(self.model_dir, f'dqn_agent_{episode}.h5'))

        # Save the final model weights
        self.agent.model.save_weights(os.path.join(self.model_dir, 'dqn_agent_final.h5'))

        return self.rewards, self.net_worths







# import numpy as np

# def train(agent, env, price_updates):
#     state = env.get_state()
#     done = False
#     total_reward = 0
#     i = 0
#     while not done and i < len(price_updates):
#         action = agent.get_action(state)  # agent selects an action
#         next_state, reward, done = env.step(action, price_updates[i])  # agent interacts with the environment
#         agent.train(state, action, reward, next_state, done)  # update Q-table
#         state = next_state
#         total_reward += reward
#         i += 1
#     return total_reward
