import random
import time
import vlc
import cv2
import numpy as np
import requests


# This is Agent(local) class for threading
class TurtleCarAgent(object):
    def __init__(self, actor, critic, optimizer, env_name, discount_factor, action_size, state_size):

        self.states = []
        self.rewards = []
        self.actions = []
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.env_name = env_name
        self.discount_factor = discount_factor
        self.action_size = action_size
        self.state_size = state_size
        self.episode = 0
        self.EPISODES = 50000
        self.scores = []
        self.rtsp = 'rtsp://192.168.1.98:8554/unicast'
        self.player = vlc.MediaPlayer(self.rtsp)
        self.player.play()
        self.frame = None
        self.current_video_time = None

    # Thread interactive with environment
    def start(self):

        while self.episode < self.EPISODES:
            score = 0
            step = 0
            state, state = self.get_state()
            while True:
                action = self.get_action(state)

                next_state, reward, done, _ = self.do_action(action)
                # reward /= 10
                score += reward
                step += 1

                action = action[0]
                self.memory(state, action, reward)

                state = next_state

                if done:
                    self.episode += 1
                    print("episode: ", self.episode, "/ score : ", score, "/ step : ", step)
                    self.scores.append(score)
                    self.train_episode(score != 500)
                    break

    def do_action(self, action):
        action = np.argmax(action)
        code = 5  # stop
        if action == 0:
            code = 4  # left
        elif action == 1:
            code = 6  # right
        elif action == 2:
            code = 8  # straight
        elif action == 3:
            code = 2  # turn down

        response = requests.get('http://192.168.1.98:5000/control?action={}'.format(code))
        if response.status_code == 200:
            serial_response = response.text.split('\r\n')
            left_sensor = float(serial_response[3])
            right_sensor = float(serial_response[4])
            status, state = self.get_state()
            done = False
            reward = 1

            if left_sensor < 20:
                reward = -1
                done = True
                requests.get('http://192.168.1.98:5000/control?action={}'.format(6))
            if right_sensor < 20:
                reward = -1
                done = True
                requests.get('http://192.168.1.98:5000/control?action={}'.format(4))

            info = {}
        else:
            status, state = self.get_state()
            reward = np.random.uniform(-10, 10)
            done = random.choice([True, False])
            info = {}
        return state, reward, done, info

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.array(self.states, dtype='float32'))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # update policy network and value network every episode
    def train_episode(self, done):
        print('Train episode: {}'.format(self.episode))
        discounted_rewards = self.discount_rewards(self.rewards, done)
        discounted_rewards = np.expand_dims(discounted_rewards, axis=1)
        # print("states size : ",len(self.states)," ", len(self.states[0]))
        # print("actions_size : ",len(self.actions))

        states = np.array(self.states, dtype='float32')

        values = self.critic.predict(states)
        # print("value : ", values.shape)
        # values = np.reshape(values, (len(values), 1))
        # print("value2 : ", values.shape)

        advantages = discounted_rewards - values

        action = np.array(self.actions)
        # print(action.shape)
        # print(advantages.shape)

        # print(states.shape, action.shape, advantages.shape)
        self.optimizer[0]([states, action, advantages])
        self.optimizer[1]([states, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        mu, sigma_sq = self.actor.predict(state)
        # sigma_sq = np.log(np.exp(sigma_sq + 1))
        epsilon = np.random.randn(self.action_size)
        action = mu + np.sqrt(sigma_sq) * epsilon
        action = np.clip(action, -2, 2)
        return action

    def get_state(self):
        while True:
            read_ret = self.player.video_take_snapshot(0, '/media/thinh/E04485F04485CA2C/snapshot.png', 96, 96)
            if read_ret == 0:
                frame = cv2.imread('/media/thinh/E04485F04485CA2C/snapshot.png')
                # frame = cv2.resize(frame, (96, 96))
                cv2.imshow('demo', frame)
                cv2.waitKey(1)
                frame = frame / 255.
                return True, frame

    def short_trial_read_rtsp(self, vcap, interval_in_s):
        # one fifth of expected interval in second for next frame
        return self.read_rtsp(vcap, wtime=interval_in_s / 5., max_nof_trial=3)

    def long_trial_read_rtsp(self, vcap, interval_in_s):
        return self.read_rtsp(vcap, wtime=interval_in_s, max_nof_trial=5)

    @staticmethod
    def read_rtsp(vcap, wtime=1., max_nof_trial=3):
        ret = False
        trial = 0
        frame = None
        while not ret and (trial < max_nof_trial):
            ret, frame = vcap.read()
            trial += 1
            if not ret:
                time.sleep(wtime)
        return ret, frame

    @staticmethod
    def grab_rtsp(vcap, wtime=1., max_nof_trial=3):
        ret = False
        trial = 0
        while not ret and (trial < max_nof_trial):
            ret = vcap.grab()
            trial += 1
            if not ret:
                time.sleep(wtime)
        return ret

    def short_trial_grab_rtsp(self, vcap, interval_in_s):
        # one fifth of expected interval in second for next frame
        return self.grab_rtsp(vcap, wtime=interval_in_s / 5., max_nof_trial=3)

    def long_trial_grab_rtsp(self, vcap, interval_in_s):
        return self.grab_rtsp(vcap, wtime=interval_in_s, max_nof_trial=5)
