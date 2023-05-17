import matplotlib.pyplot as plt
import os
import pickle

class Evaluator:

    def __init__(self, env, agent, initial_prices, price_updates, reward_type):
        self.env = env
        self.agent = agent
        self.initial_prices = initial_prices
        self.price_updates = price_updates
        self.reward_type = reward_type
        self.model_dir = f'models_{self.reward_type}'
        os.makedirs(self.model_dir, exist_ok=True)

    def evaluate(self):
        # Evaluate the agent
        evaluation_episodes = 50
        average_reward = 0
        total_trades = 0
        for _ in range(evaluation_episodes):
            self.env.reset(self.initial_prices)
            total_reward = 0
            i = 0
            while True:
                if i == len(self.price_updates):
                    break
                state = self.env.get_state()
                action = self.agent.act(state)
                _, reward, trades, done = self.env.step(action, self.price_updates.iloc[i])
                total_reward += reward
                total_trades += trades
                i += 1
                if done:
                    break
            average_reward += total_reward
        average_reward /= evaluation_episodes
        average_trades = total_trades / evaluation_episodes
        print('Average reward:', average_reward)
        print('Average trades:', average_trades)

        # Save the final model and results
        with open(os.path.join(self.model_dir, 'dqn_agent_final.pkl'), 'wb') as f:
            pickle.dump(self.agent, f)
        
        with open(os.path.join(self.model_dir, 'results.pkl'), 'wb') as f:
            pickle.dump((average_reward, average_trades), f)

        return average_reward, average_trades
    
    def evaluate_and_compare_with_benchmark(self):
        # Evaluate the agent as before
        average_reward = self.evaluate()

        # Calculate the return of the S&P 500 over the same period
        sp500_start = self.initial_prices['^GSPC']
        sp500_end = self.price_updates['^GSPC'].iloc[-1] + sp500_start
        sp500_return = (sp500_end - sp500_start) / sp500_start
        print('S&P 500 return:', sp500_return)

        # Calculate the return of the agent
        agent_start_net_worth = self.env.initial_budget
        agent_end_net_worth = self.env.get_net_worth()
        agent_return = (agent_end_net_worth - agent_start_net_worth) / agent_start_net_worth
        print('Agent return:', agent_return)

        # Compare the returns
        if agent_return > sp500_return:
            print('The agent outperformed the S&P 500.')
        else:
            print('The agent did not outperform the S&P 500.')

    def plot_results(self, rewards, net_worths):
        # Plot the rewards and net worths
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title(f'Episode Rewards - Reward System {self.reward_type}')

        plt.subplot(1, 2, 2)
        plt.plot(net_worths)
        plt.title(f'Net Worth per Episode - Reward System {self.reward_type}')
        plt.xlabel('Episode')
        plt.ylabel('Net Worth')
        plt.show()

@staticmethod
def compare_results(reward_types):
    plt.figure(figsize=(15, 10))

    average_rewards = []
    average_trades = []

    for reward_type in reward_types:
        with open(os.path.join(f'models_{reward_type}', 'results.pkl'), 'rb') as f:
            average_reward, average_trade = pickle.load(f)
            average_rewards.append(average_reward)
            average_trades.append(average_trade)

    plt.subplot(2, 1, 1)
    plt.bar(reward_types, average_rewards)
    plt.title('Average Reward')
    plt.xlabel('Reward System')
    plt.ylabel('Average Reward')

    plt.subplot(2, 1, 2)
    plt.bar(reward_types, average_trades)
    plt.title('Average Trades')
    plt.xlabel('Reward System')
    plt.ylabel('Average Trades')

    plt.show()

























# import matplotlib.pyplot as plt
# import os
# import pickle

# class Evaluator:

#     def __init__(self, env, agent, initial_prices, price_updates):
#         self.env = env
#         self.agent = agent
#         self.initial_prices = initial_prices
#         self.price_updates = price_updates
#         self.model_dir = 'models'

#     def evaluate(self):
#         # Evaluate the agent
#         evaluation_episodes = 50
#         average_reward = 0
#         for _ in range(evaluation_episodes):
#             self.env.reset(self.initial_prices)
#             total_reward = 0
#             i = 0
#             while True:
#                 if i == len(self.price_updates):
#                     break
#                 state = self.env.get_state()
#                 action = self.agent.act(state)
#                 _, reward, done = self.env.step(action, self.price_updates.iloc[i])
#                 total_reward += reward
#                 i += 1
#                 if done:
#                     break
#             average_reward += total_reward
#         average_reward /= evaluation_episodes
#         print('Average reward:', average_reward)

#         # Save the final model
#         with open(os.path.join(self.model_dir, 'dqn_agent_final.pkl'), 'wb') as f:
#             pickle.dump(self.agent, f)

#         return average_reward

#     def plot_results(self, rewards, net_worths):
#         # Plot the rewards and net worths
#         plt.figure(figsize=(15, 5))

#         plt.subplot(1, 2, 1)
#         plt.plot(rewards)
#         plt.title('Episode Rewards')

#         plt.subplot(1, 2, 2)
#         plt.plot(net_worths)
#         plt.title('Net Worth per Episode')
#         plt.xlabel('Episode')
#         plt.ylabel('Net Worth')
#         plt.show()












# import numpy as np

# def evaluate(agent, env, price_updates):
#     state = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action = agent.get_action(state)  # agent selects an action
#         next_state, reward, done = env.step(action, price_updates)  # agent interacts with the environment
#         total_reward += reward  # accumulate reward
#         state = next_state
#     return total_reward  # return total reward








# import numpy as np

# def evaluate(agent, env, price_updates):
#     state = env.get_state()
#     done = False
#     total_reward = 0
#     while not done:
#         action = agent.get_action(state)  # agent selects an action
#         next_state, reward, done = env.step(action, price_updates)  # agent interacts with the environment
#         total_reward += reward  # accumulate reward
#         state = next_state
#     return total_reward  # return total reward


