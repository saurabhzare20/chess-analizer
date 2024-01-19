
Chess Analyzer
Project Overview
Chess Analyzer is a project that employs multiple approaches to analyze and predict chess game moves. The project explores three distinct methodologies to enhance chess move predictions and gain insights into winning strategies. The three approaches include:

Neural Network Approach:

Description: Utilizing Convolutional Neural Networks (CNN) to learn and predict chess moves.
Training: Train the CNN on a dataset comprising several hundred chess games, allowing the network to learn move patterns.
Prediction: Given an initial board state, the trained CNN predicts the most likely move.
Reinforcement Learning: Fine-tune the trained parameters using reinforcement learning techniques.
Recursion Approach:

Description: Applying recursive evaluation to analyze each board position and identify potential mating threats.
Evaluation: The recursive algorithm explores potential moves, assessing their impact on the game state.
Scoring: Assign rough values to each move based on the likelihood of a favorable outcome.
Mating Threats: Detect and consider potential mating threats in the recursive evaluation.
CSV File Lookup Approach:

Description: Storing chess game lines and results in a CSV file and utilizing it to determine optimal moves.
Data Organization: Maintain a CSV file with various chess game lines and their corresponding results.
Lookup Strategy: When faced with a specific game state, search the CSV file for the most frequent winning move sequences.
Decision Making: Choose the move with the highest frequency of success as a strategy.
Project Structure
neural_network/

Contains scripts related to the neural network approach, including model training, prediction, and reinforcement learning.
recursion_approach/

Houses the recursive evaluation script, scoring functions, and mating threat detection.
csv_lookup_approach/

Includes the CSV file for storing game lines and results, along with the script for lookup strategy.
utility_functions/

Centralized location for utility functions used across different approaches, such as board representation conversion, move parsing, and data preprocessing.
Usage
Neural Network Approach:

Navigate to neural_network/ and follow the instructions in the provided scripts for model training, prediction, and reinforcement learning.
Recursion Approach:

Explore recursion_approach/ to understand and use the recursive evaluation script. Modify and extend the scoring functions as needed.
CSV File Lookup Approach:

In csv_lookup_approach/, refer to the CSV file containing game lines and results. Use the lookup script to find optimal moves based on historical data.
Contribution Guidelines
Contributions to the Chess Analyzer project are welcome! Feel free to submit issues, suggest improvements, or create pull requests.
