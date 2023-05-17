import numpy as np
import pandas as pd

class InvestmentEnvironment:
    def __init__(self, initial_budget=1000000, transaction_cost=0.0, reward_system='A'):
        self.initial_budget = initial_budget
        self.budget = initial_budget
        self.transaction_cost = transaction_cost
        self.total_returns = None
        self.returns_std_dev = None
        self.initial_prices = None
        self.current_prices = None
        self.holdings = None  # Holdings of different assets
        self.reward_system = reward_system  # Add reward system parameter
        # print(f"Reward system in environment: {self.reward_system}")

    def reset(self, initial_prices):
        assert isinstance(initial_prices, pd.Series), "initial_prices should be a pandas Series"
        self.budget = self.initial_budget
        self.initial_prices = initial_prices
        self.current_prices = initial_prices.copy()  # Assign current prices as a copy of initial prices
        self.total_returns = pd.Series(np.zeros_like(self.initial_prices), index=self.initial_prices.index)
        self.returns_std_dev = pd.Series(np.zeros_like(self.initial_prices), index=self.initial_prices.index)
        self.holdings = pd.Series(np.zeros_like(self.initial_prices), index=self.initial_prices.index)  # Reset holdings
        # print(f"Reward system in reset: {self.reward_system}")  # Print current reward system
        return self.get_state()


    def step(self, action, price_updates):
        assert isinstance(price_updates, pd.Series), "price_updates should be a pandas Series"
        # Apply transaction cost
        self.budget -= self.transaction_cost
        # Update prices
        previous_prices = self.current_prices.copy()
        self.current_prices += price_updates
        # Update returns and std dev
        returns = (self.current_prices - previous_prices) / previous_prices
        self.total_returns += returns
        self.returns_std_dev = np.sqrt(((self.returns_std_dev**2 + self.total_returns**2) / 2) - (self.total_returns / 2)**2)
        # Update budget and holdings
        self.budget -= self.current_prices[action]
        self.holdings[action] += 1  # Buy one unit of the asset
        # Calculate trades
        trades = self.current_prices - previous_prices

        # Calculate rewards
        if self.reward_system == 'A':
            sharpe_ratio = self.total_returns / self.returns_std_dev
            total_return = self.budget - self.initial_budget
            reward = sharpe_ratio.sum() + total_return  # combined reward
        else:  # Reward System B
            total_return = self.budget - self.initial_budget
            reward = total_return  # Only consider the total return

        # Update done condition
        done = self.budget < np.min(self.current_prices)
        return self.get_state(), reward, trades, done

    def get_state(self):
        if self.current_prices is None:
            # Return default state when current_prices is None
            return pd.DataFrame()
        state = self.current_prices._append(pd.Series([self.budget], index=['budget']))
        assert isinstance(state, pd.Series), "State should be a pandas Series"
        return np.array(state)

    def get_net_worth(self):
        # Calculate net worth as the sum of budget and the market value of holdings
        market_value_holdings = np.sum(self.holdings * self.current_prices)
        return self.budget + market_value_holdings





# import numpy as np
# import pandas as pd

# class InvestmentEnvironment:
#     def __init__(self, initial_budget=1000000, transaction_cost=0.0):
#         self.initial_budget = initial_budget
#         self.budget = initial_budget
#         self.transaction_cost = transaction_cost
#         self.total_returns = None
#         self.returns_std_dev = None
#         self.initial_prices = None
#         self.current_prices = None
#         self.holdings = None  # Holdings of different assets
    
#     def reset(self, initial_prices):
#         assert isinstance(initial_prices, pd.Series), "initial_prices should be a pandas Series"
#         self.budget = self.initial_budget
#         self.initial_prices = initial_prices
#         self.current_prices = initial_prices.copy()  # Assign current prices as a copy of initial prices
#         self.total_returns = pd.Series(np.zeros_like(self.initial_prices), index=self.initial_prices.index)
#         self.returns_std_dev = pd.Series(np.zeros_like(self.initial_prices), index=self.initial_prices.index)
#         self.holdings = pd.Series(np.zeros_like(self.initial_prices), index=self.initial_prices.index)  # Reset holdings
#         return self.get_state()

#     def step(self, action, price_updates):
#         assert isinstance(price_updates, pd.Series), "price_updates should be a pandas Series"
#         # Apply transaction cost
#         self.budget -= self.transaction_cost
#         # Update prices
#         previous_prices = self.current_prices.copy()
#         self.current_prices += price_updates
#         # Update returns and std dev
#         returns = (self.current_prices - previous_prices) / previous_prices
#         self.total_returns += returns
#         self.returns_std_dev = np.sqrt(((self.returns_std_dev**2 + self.total_returns**2) / 2) - (self.total_returns / 2)**2)
#         # Update budget and holdings
#         self.budget -= self.current_prices[action]
#         self.holdings[action] += 1  # Buy one unit of the asset
#         # Calculate rewards
#         sharpe_ratio = self.total_returns / self.returns_std_dev
#         total_return = self.budget - self.initial_budget
#         reward = sharpe_ratio.sum() + total_return  # combined reward
#         # Update done condition
#         done = self.budget < np.min(self.current_prices)
#         return self.get_state(), reward, done

#     def get_state(self):
#         if self.current_prices is None:
#             # Return default state when current_prices is None
#             return pd.DataFrame()
#         state = self.current_prices._append(pd.Series([self.budget], index=['budget']))
#         assert isinstance(state, pd.Series), "State should be a pandas Series"
#         print(f"Length of state in get_state: {len(state)}")  # Debug print statement
#         print(f"State shape: {state.shape}")
#         print("State in get_state:", state)
#         return np.array(state)

    
#     def get_net_worth(self):
#         # Calculate net worth as the sum of budget and the market value of holdings
#         market_value_holdings = np.sum(self.holdings * self.current_prices)
#         return self.budget + market_value_holdings

