import pandas as pd
from environment import InvestmentEnvironment
from agent import DQNAgent
from data_aggregator import DataAggregator
from train import Trainer
from evaluate import Evaluator

class RLSystem:
    def __init__(self, recent_data_path, historical_data_path, reward_type='B', window_size=5, train_test_split_ratio=0.8):
        # Load data
        self.recent_data = pd.read_csv(recent_data_path)
        self.historical_data = pd.read_csv(historical_data_path)
        self.reward_type = reward_type  # Add reward system parameter

        # Prepare data
        self.data_aggregator = DataAggregator(window_size=window_size)
        self.initial_prices, self.price_updates = self.data_aggregator.prepare_data_for_rl(self.recent_data, self.historical_data)

        # Split the data into training and testing sets
        split_index = int(len(self.price_updates) * train_test_split_ratio)
        self.initial_prices_train, self.price_updates_train = self.initial_prices[:split_index], self.price_updates[:split_index]
        self.initial_prices_test, self.price_updates_test = self.initial_prices[split_index:], self.price_updates[split_index:]

        # Initialize environment and agent
        self.env = InvestmentEnvironment(initial_budget=1000000, transaction_cost=0.0, reward_system=self.reward_type)  # Pass reward system to the environment
        self.initial_state = self.env.reset(self.initial_prices_train)
        self.agent = DQNAgent(env=self.env, state_size=len(self.initial_state), n_action=len(self.initial_prices_train))

    def keep_common_assets(self, df1, df2):
        # Keep only assets that are present in both DataFrames
        common_assets = self.initial_prices_train.index.intersection(self.price_updates_train.columns)
        self.initial_prices_train = self.initial_prices_train[common_assets]
        self.price_updates_train = self.price_updates_train[common_assets]

        common_assets_test = self.initial_prices_test.index.intersection(self.price_updates_test.columns)
        self.initial_prices_test = self.initial_prices_test[common_assets_test]
        self.price_updates_test = self.price_updates_test[common_assets_test]

        # Print out the shapes and first few rows of the dataframes
        print("Shape of initial_prices_train: ", self.initial_prices_train.shape)
        print("First few rows of initial_prices_train:\n", self.initial_prices_train.head())

        print("Shape of price_updates_train: ", self.price_updates_train.shape)
        print("First few rows of price_updates_train:\n", self.price_updates_train.head())

        print("Shape of initial_prices_test: ", self.initial_prices_test.shape)
        print("First few rows of initial_prices_test:\n", self.initial_prices_test.head())

        print("Shape of price_updates_test: ", self.price_updates_test.shape)
        print("First few rows of price_updates_test:\n", self.price_updates_test.head())

        print(f"Initial columns in df1: {df1.columns}")
        print(f"Initial columns in df2: {df2.columns}")

        common_assets = df1.columns.intersection(df2.columns)

        df1 = df1[common_assets]
        df2 = df2[common_assets]

        print(f"Common columns: {common_assets}")
        print(f"Columns in df1 after intersection: {df1.columns}")
        print(f"Columns in df2 after intersection: {df2.columns}")

        return df1, df2


    def train_and_evaluate(self):
        # Initialize Trainer and Evaluator
        trainer = Trainer(env=self.env, agent=self.agent, initial_prices=self.initial_prices_train, price_updates=self.price_updates_train)
        evaluator = Evaluator(env=self.env, agent=self.agent, initial_prices=self.initial_prices_test, price_updates=self.price_updates_test, reward_type=self.reward_type)
    
        # Print out which reward system is being used
        if self.reward_type == 'A':
            print("Training with reward system: A")
        elif self.reward_type == 'B':
            print("Training with reward system: B")
        else:
            print("Invalid reward system")
    
        # Train the agent
        self.rewards, self.net_worths = trainer.train()
    
        # Evaluate the agent
        self.average_reward, self.average_trades = evaluator.evaluate()
    
        # Evaluate the agent and compare with the S&P 500
        evaluator.evaluate_and_compare_with_benchmark()
    
        # Plot results
        evaluator.plot_results(self.rewards, self.net_worths)

    def get_results(self):
        return self.rewards, self.net_worths, self.average_reward, self.average_trades


if __name__ == "__main__":
    # Training with reward system A
    rl_system_A = RLSystem('recent_features_financial.csv', 'combined_data_features.csv', reward_type='A')
    rl_system_A.train_and_evaluate()

    # Training with reward system B
    rl_system_B = RLSystem('recent_features_financial.csv', 'combined_data_features.csv', reward_type='B')
    rl_system_B.train_and_evaluate()

    # Compare and plot the results
    evaluator_A = Evaluator(env=rl_system_A.env, agent=rl_system_A.agent, initial_prices=rl_system_A.initial_prices_test, price_updates=rl_system_A.price_updates_test, reward_type='A')
    evaluator_B = Evaluator(env=rl_system_B.env, agent=rl_system_B.agent, initial_prices=rl_system_B.initial_prices_test, price_updates=rl_system_B.price_updates_test, reward_type='B')

    # Get the results
    results_A = rl_system_A.get_results()
    results_B = rl_system_B.get_results()

    evaluator_A.plot_results(results_A[0], results_A[1])
    evaluator_B.plot_results(results_B[0], results_B[1])

    # After training and evaluating with reward system A and B
    Evaluator.compare_results(['A', 'B'])


































# import pandas as pd
# from environment import InvestmentEnvironment
# from agent import DQNAgent
# from data_aggregator import DataAggregator
# from train import Trainer
# from evaluate import Evaluator

# class RLSystem:
#     def __init__(self, recent_data_path, historical_data_path, window_size=5, train_test_split_ratio=0.8):
#         # Load data
#         self.recent_data = pd.read_csv(recent_data_path)
#         self.historical_data = pd.read_csv(historical_data_path)

#         # Prepare data
#         self.data_aggregator = DataAggregator(window_size=window_size)
#         self.initial_prices, self.price_updates = self.data_aggregator.prepare_data_for_rl(self.recent_data, self.historical_data)

#         # Split the data into training and testing sets
#         split_index = int(len(self.price_updates) * train_test_split_ratio)
#         self.initial_prices_train, self.price_updates_train = self.initial_prices[:split_index], self.price_updates[:split_index]
#         self.initial_prices_test, self.price_updates_test = self.initial_prices[split_index:], self.price_updates[split_index:]

#         # Initialize environment and agent
#         self.env = InvestmentEnvironment(initial_budget=1000000, transaction_cost=0.0)
#         self.initial_state = self.env.reset(self.initial_prices_train)
#         self.agent = DQNAgent(env=self.env, state_size=len(self.initial_state), n_action=len(self.initial_prices_train))

#     def keep_common_assets(self, df1, df2):
#         # Keep only assets that are present in both DataFrames
#         common_assets = self.initial_prices_train.index.intersection(self.price_updates_train.columns)
#         self.initial_prices_train = self.initial_prices_train[common_assets]
#         self.price_updates_train = self.price_updates_train[common_assets]

#         common_assets_test = self.initial_prices_test.index.intersection(self.price_updates_test.columns)
#         self.initial_prices_test = self.initial_prices_test[common_assets_test]
#         self.price_updates_test = self.price_updates_test[common_assets_test]

#         # Print out the shapes and first few rows of the dataframes
#         print("Shape of initial_prices_train: ", self.initial_prices_train.shape)
#         print("First few rows of initial_prices_train:\n", self.initial_prices_train.head())

#         print("Shape of price_updates_train: ", self.price_updates_train.shape)
#         print("First few rows of price_updates_train:\n", self.price_updates_train.head())

#         print("Shape of initial_prices_test: ", self.initial_prices_test.shape)
#         print("First few rows of initial_prices_test:\n", self.initial_prices_test.head())

#         print("Shape of price_updates_test: ", self.price_updates_test.shape)
#         print("First few rows of price_updates_test:\n", self.price_updates_test.head())

#         print(f"Initial columns in df1: {df1.columns}")
#         print(f"Initial columns in df2: {df2.columns}")

#         common_assets = df1.columns.intersection(df2.columns)

#         df1 = df1[common_assets]
#         df2 = df2[common_assets]

#         print(f"Common columns: {common_assets}")
#         print(f"Columns in df1 after intersection: {df1.columns}")
#         print(f"Columns in df2 after intersection: {df2.columns}")

#         return df1, df2


#     def train_and_evaluate(self):
#         # Initialize Trainer and Evaluator
#         trainer = Trainer(env=self.env, agent=self.agent, initial_prices=self.initial_prices_train, price_updates=self.price_updates_train)
#         evaluator = Evaluator(env=self.env, agent=self.agent, initial_prices=self.initial_prices_test, price_updates=self.price_updates_test)

#         # Train the agent
#         rewards, net_worths = trainer.train()

#         # Evaluate the agent
#         average_reward = evaluator.evaluate()

#         # Plot results
#         evaluator.plot_results(rewards, net_worths)


# if __name__ == "__main__":
#     rl_system = RLSystem('recent_features_financial.csv', 'combined_data_features.csv')
#     rl_system.keep_common_assets(rl_system.recent_data, rl_system.historical_data)
#     rl_system.train_and_evaluate()


























# from environment import InvestmentEnvironment
# from agent import DQNAgent
# from data_aggregator import DataAggregator
# from train import Trainer
# from evaluate import Evaluator
# import pandas as pd

# # Load data
# recent_data = pd.read_csv('recent_features_financial.csv')
# historical_data = pd.read_csv('combined_data_features.csv')

# # Prepare data
# data_aggregator = DataAggregator(window_size=5)
# initial_prices, price_updates = data_aggregator.prepare_data_for_rl(recent_data, historical_data)

# # Split the data into training and testing sets
# train_test_split_ratio = 0.8  # Adjust as necessary
# split_index = int(len(price_updates) * train_test_split_ratio)
# initial_prices_train, price_updates_train = initial_prices[:split_index], price_updates[:split_index]
# initial_prices_test, price_updates_test = initial_prices[split_index:], price_updates[split_index:]

# # Keep only assets that are present in both DataFrames
# common_assets = initial_prices_train.index.intersection(price_updates_train.columns)
# initial_prices_train = initial_prices_train[common_assets]
# price_updates_train = price_updates_train[common_assets]

# common_assets_test = initial_prices_test.index.intersection(price_updates_test.columns)
# initial_prices_test = initial_prices_test[common_assets_test]
# price_updates_test = price_updates_test[common_assets_test]

# # Initialize environment and agent
# env = InvestmentEnvironment(initial_budget=1000000, transaction_cost=0.0)
# initial_state = env.reset(initial_prices_train)
# agent = DQNAgent(env=env, state_size=len(initial_state), n_action=len(initial_prices_train))

# # Initialize Trainer and Evaluator
# trainer = Trainer(env=env, agent=agent, initial_prices=initial_prices_train, price_updates=price_updates_train)
# evaluator = Evaluator(env=env, agent=agent, initial_prices=initial_prices_test, price_updates=price_updates_test)

# # Train the agent
# rewards, net_worths = trainer.train()

# # Evaluate the agent
# average_reward = evaluator.evaluate()

# # Plot results
# evaluator.plot_results(rewards, net_worths)











# import pandas as pd
# from environment import InvestmentEnvironment
# from agent import DQNAgent  # Replace with the correct agent class from your implementation
# from data_aggregator import DataAggregator
# import os
# import pickle
# import matplotlib.pyplot as plt

# # Load data
# recent_data = pd.read_csv('recent_features_financial.csv')
# historical_data = pd.read_csv('combined_data_features.csv')

# # Print first five rows of loaded data
# print(f'Recent data:', recent_data.head())
# print(f'\nHistorical data:', historical_data.head())

# # Prepare data
# data_aggregator = DataAggregator(window_size=5)
# initial_prices, price_updates = data_aggregator.prepare_data_for_rl(recent_data, historical_data)

# # Split the data into training and testing sets
# train_test_split_ratio = 0.8  # Adjust as necessary
# split_index = int(len(price_updates) * train_test_split_ratio)
# initial_prices_train, price_updates_train = initial_prices[:split_index], price_updates[:split_index]
# initial_prices_test, price_updates_test = initial_prices[split_index:], price_updates[split_index:]

# # Keep only assets that are present in both DataFrames
# common_assets = initial_prices.index.intersection(price_updates.columns)
# initial_prices = initial_prices[common_assets]
# price_updates = price_updates[common_assets]

# # Print the number of assets after filtering
# print("Number of assets after filtering:", len(common_assets))

# # Initialize environment and agent
# env = InvestmentEnvironment(initial_budget=1000000, transaction_cost=0.0)
# initial_state = env.reset(initial_prices)
# agent = DQNAgent(env=env, state_size=len(initial_state), n_action=len(initial_prices))

# # Define the directory for saving the model
# model_dir = 'models'
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

# # Termination condition based on agent's performance
# no_improvement_threshold = 50  # Stop training after 50 episodes without improvement
# best_reward = -float('inf')
# episodes_without_improvement = 0

# # Lists to store results
# rewards = []
# net_worths = []

# # Train the agent
# n_episode = 1000
# for episode in range(n_episode):
#     env.reset(initial_prices)
#     total_reward = 0
#     i = 0
#     while True:
#         if i == len(price_updates):  # Break the loop when 'i' equals the length of 'price_updates'
#             break
#         state = env.get_state()
#         action = agent.act(state)    # action = agent.get_action(state) replace with this snippet if we change back to previous agent
#         next_state, reward, done = env.step(action, price_updates.iloc[i])
#         agent.update(state, action, reward, next_state, done)
#         total_reward += reward
#         i += 1
#         if done:
#             break

#     rewards.append(total_reward)
#     net_worths.append(env.get_net_worth())
#     print(f'Episode: {episode}, Total Reward: {total_reward}, Net Worth: {env.get_net_worth()}')
    
#     if total_reward > best_reward:
#         best_reward = total_reward
#         episodes_without_improvement = 0
#     else:
#         episodes_without_improvement += 1

#     if episodes_without_improvement >= no_improvement_threshold:
#         print(f'No improvement in {no_improvement_threshold} episodes, stopping training...')
#         break

#     # Save the model weights every 100 episodes
#     if episode % 100 == 0:
#         agent.model.save_weights(os.path.join(model_dir, f'dqn_agent_{episode}.h5'))

# # Save the final model weights
# agent.model.save_weights(os.path.join(model_dir, 'dqn_agent_final.h5'))

# # Evaluate the agent
# evaluation_episodes = 50
# average_reward = 0
# for _ in range(evaluation_episodes):
#     env.reset(initial_prices_test)
#     total_reward = 0
#     i = 0
#     while True:
#         if i == len(price_updates_test):  # Break the loop when 'i' equals the length of 'price_updates_test'
#             break
#         state = env.get_state()
#         action = agent.act(state)  # action = agent.get_action(state)
#         _, reward, done = env.step(action, price_updates_test.iloc[i])
#         total_reward += reward
#         i += 1
#         if done:
#             break
#     average_reward += total_reward
# average_reward /= evaluation_episodes
# print('Average reward:', average_reward)

# # Save the final model
# with open(os.path.join(model_dir, 'dqn_agent_final.pkl'), 'wb') as f:
#     pickle.dump(agent, f)

# # Plot the rewards and net worths
# plt.figure(figsize=(15, 5))

# plt.subplot(1, 2, 1)
# plt.plot(rewards)
# plt.title('Episode Rewards')

# plt.subplot(1, 2, 2)
# plt.plot(net_worths)
# plt.title('Net Worth')

# plt.show()











# import pandas as pd
# from environment import InvestmentEnvironment
# from agent import DQNAgent  # Replace with the correct agent class from your implementation
# from data_aggregator import DataAggregator
# import os
# import pickle

# # Load data
# recent_data = pd.read_csv('recent_features_financial.csv')
# historical_data = pd.read_csv('combined_data_features.csv')

# # Print first five rows of loaded data
# print(f'Recent data:', recent_data.head())
# print(f'\nHistorical data:', historical_data.head())

# # Prepare data
# data_aggregator = DataAggregator(window_size=5)
# initial_prices, price_updates = data_aggregator.prepare_data_for_rl(recent_data, historical_data)

# # Keep only assets that are present in both DataFrames
# common_assets = initial_prices.index.intersection(price_updates.columns)
# initial_prices = initial_prices[common_assets]
# price_updates = price_updates[common_assets]

# # Print the number of assets after filtering
# print("Number of assets after filtering:", len(common_assets))

# # Initialize environment and agent
# env = InvestmentEnvironment(initial_budget=1000000, transaction_cost=0.0)
# initial_state = env.reset(initial_prices)
# agent = DQNAgent(env=env, state_size=len(initial_state), n_action=len(initial_prices))

# # Define the directory for saving the model
# model_dir = 'models'
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

# # Train the agent
# n_episode = 1000
# for episode in range(n_episode):
#     env.reset(initial_prices)
#     total_reward = 0
#     i = 0
#     while True:
#         if i == len(price_updates):  # Break the loop when 'i' equals the length of 'price_updates'
#             break
#         state = env.get_state()
#         action = agent.act(state)    # action = agent.get_action(state) replace with this snippet if we change back to previous agent
#         next_state, reward, done = env.step(action, price_updates.iloc[i])
#         agent.update(state, action, reward, next_state, done)
#         total_reward += reward
#         i += 1
#         if done:
#             break


#     print(f'Episode: {episode}, Total Reward: {total_reward}, Net Worth: {env.get_net_worth()}')

#     # Save the model weights every 100 episodes
#     if episode % 100 == 0:
#         agent.model.save_weights(os.path.join(model_dir, f'dqn_agent_{episode}.h5'))

# # Save the final model weights
# agent.model.save_weights(os.path.join(model_dir, 'dqn_agent_final.h5'))

# # Evaluate the agent
# average_reward = 0
# for _ in range(100):
#     env.reset(initial_prices)
#     total_reward = 0
#     i = 0
#     while True:
#         if i == len(price_updates):  # Break the loop when 'i' equals the length of 'price_updates'
#             break
#         state = env.get_state()
#         action = agent.act(state)   # action = agent.get_action(state)
#         _, reward, done = env.step(action, price_updates.iloc[i])
#         total_reward += reward
#         i += 1
#         if done:
#             break
#     average_reward += total_reward
# average_reward /= 100
# print('Average reward:', average_reward)

# # Save the final model
# with open(os.path.join(model_dir, 'dqn_agent_final.pkl'), 'wb') as f:
#     pickle.dump(agent, f)














# import pandas as pd
# from environment import InvestmentEnvironment
# from agent import DQNAgent  # change this to the correct agent class
# from data_aggregator import DataAggregator
# import os
# import pickle

# # Load data
# recent_data = pd.read_csv('recent_features_financial.csv')
# historical_data = pd.read_csv('combined_data_features.csv')

# # Prepare data
# data_aggregator = DataAggregator(window_size=5)
# initial_prices, price_updates = data_aggregator.prepare_data_for_rl(recent_data, historical_data)

# # Initialize environment and agent
# env = InvestmentEnvironment(initial_budget=1000, transaction_cost=0.0)
# agent = DQNAgent(n_action=len(initial_prices), state_size=len(env.get_state()))  


# # Define the directory for saving the model
# model_dir = 'models'
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

# # Train the agent
# n_episode = 1000
# for episode in range(n_episode):
#     env.reset(initial_prices)
#     total_reward = 0
#     i = 0
#     while True:
#         state = env.get_state()
#         action = agent.get_action(state)
#         next_state, reward, done = env.step(action, price_updates.iloc[i])
#         agent.update(state, action, reward, next_state, done)
#         total_reward += reward
#         i += 1
#         if done:
#             break

#     print(f'Episode: {episode}, Total Reward: {total_reward}, Net Worth: {env.get_net_worth()}')

#     # Save the model weights every 100 episodes
#     if episode % 100 == 0:
#         agent.model.save_weights(os.path.join(model_dir, f'dqn_agent_{episode}.h5'))

#     # Save the final model weights
#     agent.model.save_weights(os.path.join(model_dir, 'dqn_agent_final.h5'))


# # Evaluate the agent
# average_reward = 0
# for _ in range(100):
#     env.reset(initial_prices)
#     total_reward = 0
#     i = 0
#     while True:
#         state = env.get_state()
#         action = agent.get_action(state)
#         _, reward, done = env.step(action, price_updates[i])
#         total_reward += reward
#         i += 1
#         if done:
#             break
#     average_reward += total_reward
# average_reward /= 100
# print('Average reward:', average_reward)

# # Save the final model
# with open(os.path.join(model_dir, 'dqn_agent_final.pkl'), 'wb') as f:
#     pickle.dump(agent, f)





















# # import pandas as pd
# # from environment import InvestmentEnvironment
# # from agent import QLearningAgent
# # from train import train
# # from evaluate import evaluate
# # from data_aggregator import DataAggregator
# # import os
# # import pickle

# # # Load data
# # recent_data = pd.read_csv('recent_features_financial.csv')
# # historical_data = pd.read_csv('combined_data_features.csv')

# # # Prepare data
# # data_aggregator = DataAggregator(window_size=5, lookback=10, n_features=5)
# # initial_prices, price_updates = data_aggregator.prepare_data_for_rl(recent_data, historical_data)

# # # Initialize environment and agent
# # env = InvestmentEnvironment(initial_budget=1000, initial_prices=initial_prices)
# # agent = QLearningAgent(n_action=5, n_state=10)  # Here you need to decide the number of discrete states

# # # Define the directory for saving the model
# # model_dir = 'models'
# # if not os.path.exists(model_dir):
# #     os.makedirs(model_dir)

# # # Train the agent
# # n_episode = 1000
# # for episode in range(n_episode):
# #     train(agent, env, price_updates, n_episode)
    
# #     # Save the model every 100 episodes
# #     if episode % 100 == 0:
# #         with open(os.path.join(model_dir, f'q_learning_agent_{episode}.pkl'), 'wb') as f:
# #             pickle.dump(agent, f)

# # # Evaluate the agent
# # average_reward = evaluate(agent, env, n_episode=100)
# # print('Average reward:', average_reward)

# # # Save the final model
# # with open(os.path.join(model_dir, 'q_learning_agent_final.pkl'), 'wb') as f:
# #     pickle.dump(agent, f)
