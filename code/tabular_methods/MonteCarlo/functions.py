import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  



def MonteCarloLearnStateValueFunction(env,stateNumber,numberOfEpisodes,discountRate):

    sumReturnForEveryState=np.zeros(stateNumber)
    numberVisitsForEveryState=np.zeros(stateNumber)
    valueFunctionEstimate=np.zeros(stateNumber)

    for indexEpisode in range(numberOfEpisodes):
        visitedStatesInEpisode=[]
        rewardInVisitedState=[]
        (currentState,prob)=env.reset()
        visitedStatesInEpisode.append(currentState)
        if indexEpisode%100 == 0:
            print("Simulating episode {}".format(indexEpisode))

        while True:
            randomAction= env.action_space.sample()
            (currentState, currentReward, terminalState,_,_) = env.step(randomAction)         
            rewardInVisitedState.append(currentReward)
            if not terminalState:
                visitedStatesInEpisode.append(currentState)   
            else: 
                break

        numberOfVisitedStates=len(visitedStatesInEpisode)
        Gt=0
        for indexCurrentState in range(numberOfVisitedStates-1,-1,-1):
                
            stateTmp=visitedStatesInEpisode[indexCurrentState] 
            returnTmp=rewardInVisitedState[indexCurrentState]
            Gt=discountRate*Gt+returnTmp
            if stateTmp not in visitedStatesInEpisode[0:indexCurrentState]:
                numberVisitsForEveryState[stateTmp]=numberVisitsForEveryState[stateTmp]+1
                sumReturnForEveryState[stateTmp]=sumReturnForEveryState[stateTmp]+Gt

    for indexSum in range(stateNumber):
        if numberVisitsForEveryState[indexSum] !=0:
            valueFunctionEstimate[indexSum]=sumReturnForEveryState[indexSum]/numberVisitsForEveryState[indexSum]
        
    return valueFunctionEstimate

def evaluatePolicy(env,valueFunctionVector,policy,discountRate,maxNumberOfIterations,convergenceTolerance):
    convergenceTrack=[]
    for iterations in range(maxNumberOfIterations):
        convergenceTrack.append(np.linalg.norm(valueFunctionVector,2))
        valueFunctionVectorNextIteration=np.zeros(env.observation_space.n)
        for state in env.P:
            outerSum=0
            for action in env.P[state]:
                innerSum=0
                for probability, nextState, reward, isTerminalState in env.P[state][action]:
                    innerSum=innerSum+ probability*(reward+discountRate*valueFunctionVector[nextState])
                outerSum=outerSum+policy[state,action]*innerSum
            valueFunctionVectorNextIteration[state]=outerSum
        if(np.max(np.abs(valueFunctionVectorNextIteration-valueFunctionVector))<convergenceTolerance):
            valueFunctionVector=valueFunctionVectorNextIteration
            print('Iterative policy evaluation algorithm converged!')
            break
        valueFunctionVector=valueFunctionVectorNextIteration       
    return valueFunctionVector
       

def derive_policy(env, V, discountRate):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        best_action = None
        best_value = float('-inf')
        for a in range(env.action_space.n):
            action_value = 0
            for prob, next_state, reward, done in env.P[s][a]:
                action_value += prob * (reward + discountRate * V[next_state] * (not done))
            if action_value > best_value:
                best_value = action_value
                best_action = a
        policy[s] = best_action
    return policy

def evaluate_policy_performance(env, policy, num_episodes):
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()[0]
        done = False
        while not done:
            action = policy[state]
            state, reward, done, prob, _ = env.step(action)
            if done:
                total_reward += reward
    return total_reward / num_episodes 