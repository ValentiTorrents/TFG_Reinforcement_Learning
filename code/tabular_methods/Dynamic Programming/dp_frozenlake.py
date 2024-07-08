import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import copy

def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])#formula pag 75
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V

def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.observation_space.n, env.action_space.n]) / env.action_space.n
    for s in range(env.observation_space.n):
        q = q_from_v(env, V, s, gamma)
        
        # OPTION 1: construct a deterministic policy 
        # policy[s][np.argmax(q)] = 1
        
        # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
        best_a = np.argwhere(q==np.max(q)).flatten()
        policy[s] = np.sum([np.eye(env.action_space.n)[i] for i in best_a], axis=0)/len(best_a)
        
    return policy

def policy_iteration(env, gamma=1, theta=1e-8):
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V)
        
        # OPTION 1: stop if the policy is unchanged after an improvement step
        if (new_policy == policy).all():
            break;
        
        # OPTION 2: stop if the value function estimates for successive policies has converged
        # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
        #    break;
        
        policy = copy.copy(new_policy)
    return policy, V

def plot_value(values):
    V_sq = np.reshape(values, (4,4))
    # plot the state-value function
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(V_sq, cmap='cool')
    for (j,i),label in np.ndenumerate(V_sq):
        ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
    plt.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')
    plt.title('State-Value Function')
    plt.show()

def run_episode(env, action_list):
    state = env.reset()[0]
    done = False
    step_count = 0
    while not done:
        action = action_list[state]
        next_state, reward, done, truncated, _ = env.step(action)
        step_count += 1
        if done:
            break
        state = next_state
    return reward, step_count


env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)#can add render_mode="human"
random_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n

V = policy_evaluation(env, random_policy)

#plot_value(V)



Q = np.zeros([env.observation_space.n, env.action_space.n])
for s in range(env.observation_space.n):
    Q[s] = q_from_v(env, V, s)
print("Action-Value Function:")
print(Q)


# obtain the optimal policy and optimal state-value function
policy_pi, V_pi = policy_iteration(env)

# print the optimal policy
print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
print(policy_pi,"\n")


#plot_value(V_pi)

action_list_q = []
for s in policy_pi:
    max_value = np.max(s)
    max_position = np.argmax(s)
    #max_position = np.where(s == max_value)[0]
    #max_position = np.random.choice(max_positions)
    action_list_q.append(max_position)
print(action_list_q)

action_list_pi = []
for s in policy_pi:
    max_value = np.max(s)
    max_position = np.argmax(s)
    #max_position = np.where(s == max_value)[0]
    #max_position = np.random.choice(max_positions)
    action_list_pi.append(max_position)
print(action_list_pi)

# won = False
# terminated = False      
# truncated = False 
# while won==False:
#     state = env.reset()[0]
#     terminated = False      
#     truncated = False
#     while(not terminated and not truncated):
#         action = action_list_pi[state] # actions: 0=left,1=down,2=right,3=up
#         new_state,reward,terminated,truncated,_ = env.step(action)
#         state = new_state        
#         if(state == 15):
#             won=True

episodes=10000

results_pi = []
steps_pi = []
for _ in range(episodes):
    rewards, steps = run_episode(env, action_list_pi)
    results_pi.append(rewards)
    steps_pi.append(steps)
count_pi = results_pi.count(1.0)
avg_step_pi = sum(steps_pi)/len(steps_pi)
print("Pi has succeeded: ", count_pi," times with an avg of:", avg_step_pi, " steps.")

# results_q = []
# steps_q = []
# for _ in range(episodes):
#     rewards, steps = run_episode(env, action_list_q)
#     results_q.append(rewards)
#     steps_q.append(steps)
# count_q = results_q.count(1.0)
# avg_step_q = sum(steps_q)/len(steps_q)
# print("Q has succeeded: ", count_q," times with an avg of:", avg_step_q, " steps.")

env.close()