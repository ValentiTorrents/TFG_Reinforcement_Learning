import gymnasium as gym
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Initialize environment and parameters
env = gym.make('CartPole-v1')
obs_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Hyperparameters
alpha = 0.001  # Learning rate
gamma = 0.95  # Discount factor
num_episodes = 5000  # Number of episodes for training
epsilon = 0.1  # Exploration rate for epsilon-greedy policy

# Initialize weights for linear value function approximation
weights = np.random.rand((obs_space + 1) * (obs_space + 2) // 2)  # For PolynomialFeatures of degree 2
poly = PolynomialFeatures(degree=2)

# Feature function using polynomial features
def feature_function(state):
    return poly.fit_transform([state])[0]

# Value function as a linear combination of features
def value_function(state, weights):
    features = feature_function(state)
    return np.dot(weights, features)

# Update weights using gradient Monte Carlo
def update_weights(weights, states, returns, alpha):
    for t in range(len(states)):
        features = feature_function(states[t])
        value_estimate = np.dot(weights, features)
        gradient = features
        weights += alpha * (returns[t] - value_estimate) * gradient
    return weights

# Generate a single episode using a given policy
def generate_episode(env, policy):
    episode = []
    state = env.reset()[0]
    done = False

    while not done:
        action = policy(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, reward))
        state = next_state

    return episode

# Calculate returns for an episode
def calculate_returns(episode, gamma):
    returns = []
    G = 0
    for state, reward in reversed(episode):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

# Epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        return 0 if value_function(state, weights) < 0 else 1  # Exploitation (assuming binary action space)

# Generate multiple episodes for evaluation
def generate_multiple_episodes(env, num_episodes, training=True):
    episodes = []
    for _ in range(num_episodes):
        episode = generate_episode(env, epsilon_greedy_policy)
        episodes.append(episode)
        # if training == False:
        #     print("end of episode")
    return episodes

# Calculate true returns for the initial state
def calculate_initial_state_returns(episodes, gamma):
    initial_state_returns = []
    for episode in episodes:
        _, rewards = zip(*episode)
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
        initial_state_returns.append(G)
    return initial_state_returns

# Calculate mean squared error
def mean_squared_error(estimated_value, true_values):
    true_values = np.array(true_values)
    mse = np.mean((true_values - estimated_value) ** 2)
    return mse

# Training loop
for episode in range(num_episodes):
    episode_data = generate_episode(env, epsilon_greedy_policy)
    states, rewards = zip(*episode_data)
    returns = calculate_returns(episode_data, gamma)
    weights = update_weights(weights, states, returns, alpha)

    # if episode % 1000 == 0:
    #     print(f"Episode {episode}, Weights: {weights}")

print("Training completed.")
print(f"Episode {episode}, Weights: {weights}")

env = gym.make('CartPole-v1', render_mode="human")
# Estimate the value of the initial state
state = env.reset()[0]
estimated_value = value_function(state, weights)

# Generate evaluation episodes
num_eval_episodes = 100
eval_episodes = generate_multiple_episodes(env, num_eval_episodes, training=False)

# # Calculate true returns for the initial state
# true_returns = calculate_initial_state_returns(eval_episodes, gamma)

# # Calculate MSE
# mse = mean_squared_error(estimated_value, true_returns)

# print(f"Estimated Value of Initial State: {estimated_value}")
# print(f"Mean Squared Error: {mse}")
