import threading
import numpy as np
import tensorflow as tf
import time
import os
from keras.layers import Dense, Input, Lambda, Conv2D, BatchNormalization, MaxPool2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

from env import TurtleCarAgent

# global variables for threading
episode = 0
scores = []

EPISODES = 2000

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)


# This is A3C(Asynchronous Advantage Actor Critic) agent(global) for the Cartpole
# In this example, we use A3C algorithm
class A3CAgent:
    def __init__(self, state_size, action_size, env_name):
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # get gym environment name
        self.env_name = env_name

        # these are hyper parameters for the A3C
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.discount_factor = .9
        self.hidden1, self.hidden2 = 512, 128
        self.threads = 8

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        state = Input(batch_shape=self.state_size)
        actor_input = Conv2D(16, kernel_size=(3, 3), activation='relu')(state)
        actor_hidden = MaxPool2D()(actor_input)
        actor_hidden = Conv2D(8, kernel_size=(3, 3), activation='relu')(actor_hidden)
        actor_hidden = MaxPool2D()(actor_hidden)
        actor_hidden = Flatten()(actor_hidden)
        actor_hidden = Dense(self.hidden1, activation='relu')(actor_hidden)
        actor_hidden = Dense(self.hidden2, activation='relu')(actor_hidden)
        mu_0 = Dense(self.action_size, activation='tanh')(actor_hidden)
        sigma_0 = Dense(self.action_size, activation='softplus')(actor_hidden)

        mu = Lambda(lambda x: x * 2)(mu_0)
        sigma = Lambda(lambda x: x + 0.0001)(sigma_0)

        critic_input = Conv2D(16, kernel_size=(3, 3), activation='relu')(state)
        value_hidden = MaxPool2D()(critic_input)
        value_hidden = Conv2D(8, kernel_size=(3, 3), activation='relu')(value_hidden)
        value_hidden = MaxPool2D()(value_hidden)
        value_hidden = Flatten()(value_hidden)
        value_hidden = Dense(self.hidden1, input_dim=self.state_size, activation='relu')(value_hidden)
        value_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(value_hidden)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=(mu, sigma))
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    # make loss function for Policy Gradient
    # [log(action probability) * advantages] will be input for the back prop
    # we add entropy of action probability to loss
    def actor_optimizer(self):
        action = K.placeholder(shape=(None, 5))
        advantages = K.placeholder(shape=(None, 1))

        # mu = K.placeholder(shape=(None, self.action_size))
        # sigma_sq = K.placeholder(shape=(None, self.action_size))

        mu, sigma_sq = self.actor.output

        pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))

        exp_v = log_pdf * advantages

        exp_v = K.sum(exp_v + 0.01 * entropy)
        actor_loss = -exp_v

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)

        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, 1))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    # make agents(local) and start training
    def train(self):
        # self.load_model('./save_model/cartpole_a3c.h5')
        agent = TurtleCarAgent(
            self.actor, self.critic, self.optimizer,
            self.env_name, self.discount_factor,
            self.action_size, self.state_size)
        agent.start()

        while True:
            time.sleep(20)

            plot = scores[:]
            # pylab.plot(range(len(plot)), plot, 'b')
            # pylab.savefig("./save_graph/cartpole_a3c.png")

            self.save_model('./save_model/cartpole_a3c.h5')

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")


if __name__ == "__main__":
    # env_name = 'CartPole-v1'
    env_name = 'Turtle-V0'
    action_size = 5
    state_size = (None, 96, 96, 3)
    global_agent = A3CAgent(state_size, action_size, env_name)
    global_agent.train()
