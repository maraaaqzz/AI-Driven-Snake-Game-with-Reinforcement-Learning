# AI-Driven-Snake-Game-with-Reinforcement-Learning


## Features

- **Reinforcement Learning**: The AI uses Q-learning to make decisions and learn from game states.
- **Neural Network**: A deep neural network predicts the best moves based on the current game state.
- **Experience Replay**: The agent stores past experiences and reuses them to improve learning.
- **Epsilon-Greedy Strategy**: Balances exploration (trying new moves) and exploitation (using known good moves).
- **Dynamic Gameplay**: The AI continuously improves its gameplay by learning from its successes and failures.

## How It Works

1. **Game Environment**: The Snake game environment is built using Pygame.
2. **State Representation**: The current state of the game (snake's position, direction, food location, etc.) is captured and fed into the neural network.
3. **Action Prediction**: The neural network predicts the best action (move left, right, or straight) based on the current state.
4. **Reward System**: The agent receives rewards for eating food and penalties for collisions.
5. **Training**: The AI is trained using experience replay and a loss function to minimize errors in predictions.
