import gym
import tensorflow as tf
import numpy as np

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v0")
states = env.observation_space.shape[0]
actions = env.action_space.n


def build_model(states,actions):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(1,states)))
    model.add(tf.keras.layers.Dense(units=64,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=32,activation=tf.nn.tanh))
    model.add(tf.keras.layers.Dense(units=actions,activation="linear"))

    return model

    
def build_agent(model,actions):

    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000,window_length=1)
    dqn = DQNAgent(model=model,memory=memory,policy=policy,nb_actions=actions,nb_steps_warmup=500,target_model_update=1e-2)

    return dqn
