import cartpole
import gym
import random
import tensorflow as tf
import numpy as np
import os 

env = gym.make("CartPole-v0")
states = env.observation_space.shape[0]
actions = env.action_space.n

model = cartpole.build_model(states,actions)
dqn = cartpole.build_agent(model,actions)

dqn.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),metrics=["mae"])
dqn.fit(env,nb_steps=50000,visualize=True,verbose=2)

scores = dqn.test(env,nb_episodes=100,visualize=False)
print("Score: {}".format(np.mean(scores.history["episode_reward"])))

dqn.save_weights(os.getcwd()+"/model/weights.h5f",overwrite=True)

#Reload model
env = gym.make("CartPole-v0")
states = env.observation_space.shape[0]
actions = env.action_space.n

model = cartpole.build_model(states,actions)
dqn = cartpole.build_agent(model,actions)
dqn.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),metrics=["mae"])

dqn.load_weights(os.getcwd()+"/model/weights.h5f")
dqn.test(env,nb_episodes=10,visualize=True)


"""
episode = 10

for episode in range(1,episode+1):

    state = env.reset()
    done = False
    score = 0

    while not done:

        env.render()
        action = random.choice([0,1])
        n_state,reward,done,info = env.step(action)
        score += reward

    print("Episode:{} Score: {}".format(episode,score))

"""
