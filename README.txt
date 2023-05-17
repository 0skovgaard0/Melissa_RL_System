Project Description: RL-based Financial Asset Trading System

Project Overview:
The goal of this project is to build a reinforcement learning (RL) system for trading financial assets. The RL agent is trained to make investment decisions and maximize returns using a fictional budget of $1 million. The project utilizes historical assets data collected from 1970 to 2023, with a small test set representing the most recent month. The project focuses on the adjusted closing price ('adjclose') as a key indicator for the RL model's learning.

Key Dependencies:

Python 3.x
TensorFlow/Keras library
pandas
NumPy
matplotlib
Instructions for Use:

Clone the repository from GitHub.
Install the required dependencies listed above.
Update CSV file paths and other settings based on your storage and configuration.
Execute the following scripts in the specified order:
assets.py:

This script collects financial data from Yahoo Finance and macroeconomic data from various sources.
It cleans, normalizes, and organizes the data into a readable long-format.
Run the script to gather and preprocess the necessary data for the RL system.
exchange.py:

This script fetches exchange rate data for Yen and Euro against USD from CSV files and Yahoo Finance.
It converts values from Yen or Euro to USD in the gathered data.
Execute the script to ensure currency conversion for accurate comparison.
merged_data.py:

This script loads the preprocessed CSV files of financial and macroeconomic datasets.
It merges the datasets into a single comprehensive dataset for each category.
The script provides details on the shape of the merged datasets, missing values, and unique asset names.
Run the script to merge and save the filtered financial data and the merged macroeconomic data.
feature_extraction.py:

This script contains two classes: 'DataPreprocessor' and 'DataPreparation'.
'DataPreprocessor' calculates rolling mean, momentum, and volatility of given data.
'DataPreparation' applies PCA, creates lookback data, and prepares the data for model training.
Use this script to preprocess and prepare data with specific calculations and transformations.
data_aggregator.py:

This script collects data for various financial assets, cleans and normalizes it, and aggregates it into a single DataFrame.
It preprocesses the data and prepares it for reinforcement learning.
Execute this script to gather, clean, normalize, and prepare the data for the RL system.
RL System:

The following scripts are responsible for setting up, training, and evaluating the RL-based trading system:
RL_main.py:

This script manages the main components of the RL system.
It loads the preprocessed data, prepares it, trains the RL agent, and evaluates its performance.
Run this script to set up and manage the RL system.
agent.py:

This script defines the DQNAgent class, which learns and makes decisions based on feedback from the environment.
The agent is the main character of the RL system, learning and making decisions based on outcomes.
environment.py:

This script models the investment environment, including initial budget, transaction cost, and asset prices.
It defines the possible actions and their effects on the agent's state.
Use this script to modify the environment characteristics and actions.
train.py:

This script defines the training process for the RL agent.
It specifies how the agent learns from its actions and rewards.
Modify this script to adjust the training parameters if necessary.
evaluate.py:

This script evaluates the performance of the trained RL agent and visualizes the results.
Execute this script to analyze and visualize the agent's performance.
Additional Reflections and Improvements:
The 'README.txt' file includes reflections, possible improvements, and other issues encountered during the project. Remember to update CSV file paths and other settings based on your storage and configuration. If you have any questions, please reach out for assistance. Good luck with your project!

Please note that these instructions assume a basic understanding of Python programming, data preprocessing, and reinforcement learning concepts.