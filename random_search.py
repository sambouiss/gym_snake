import numpy as np
import gym, gym_snake
softmax = lambda x: np.exp(x)/np.sum(np.exp(x))
env = gym.make('snake-v0')
parameters1 = 2*np.random.rand(3,32*32+1)-1
def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = np.argmax(np.max(parameters[0].dot(observation)+parameters[1],0))
        observation, reward, done, info = env.step(action)
        totalreward = reward
        if done:
            break
    return totalreward

bestparams = ((2*np.random.rand(3,32*32+1)-1),(2*np.random.rand(3)-1))
bestreward = 0  
noise = .1
for _ in range(10000):  
    parameters = (bestparams[0]+noise*(2*np.random.rand(3,32*32+1)-1),bestparams[1]+noise*(2*np.random.rand(3)-1))
    reward = 0
    for j in range(10):
    	reward += run_episode(env,parameters)
    reward = reward/10
    if reward > bestreward:
        bestreward = reward
        print(_, bestreward)
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward >=  50:
            break

print(bestreward)
def run_episode_with_display(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        env.render()
        action = np.argmax(np.max(parameters[0].dot(observation)+parameters[1],0))
        observation, reward, done, info = env.step(action)
        totalreward = reward
        if done:
            break


    return totalreward

print(run_episode_with_display(env,bestparams))