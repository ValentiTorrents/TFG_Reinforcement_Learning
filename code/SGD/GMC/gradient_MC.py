import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Define the neural network for value function approximation
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize environment
env = gym.make('CartPole-v1')
#dimension of each state
state_dim = env.observation_space.shape[0]
#initialize network v(St, W)
value_net = ValueNetwork(state_dim)
#we specify the learning rate lr
optimizer = optim.Adam(value_net.parameters(), lr=0.01)

# Hyperparameters
gamma = 0.99

# Training loop
num_episodes = 500
for episode in range(num_episodes):
    state = env.reset()[0]
    terminated = False

    while not terminated:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = env.action_space.sample()  # Random policy for simplicity
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Compute TD error
        with torch.no_grad():
            target = reward + gamma * value_net(next_state_tensor) * (1 - terminated)
        prediction = value_net(state_tensor)
        td_error = target - prediction
        
        # Update network
        loss = td_error.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    print(f'Episode {episode + 1}, Loss: {loss.item()}')