import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Lambda
from keras.optimizers import adam_v2
import numpy as np
import matplotlib.pyplot as plt


# actor network
class Actor(Model):

    def __init__(self, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')

        # continuous head
        self.h_discrete = Dense(16, activation='relu')
        self.mu = Dense(1, activation='tanh')
        self.std = Dense(1, activation='softplus')

        # discrete head
        self.h_continuous = Dense(16, activation='relu')
        self.discrete_pdf = Dense(2, activation='softmax')

    def call(self, state):
        # common
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)

        # continuous
        disc = self.h_discrete(x)
        mu = self.mu(disc)
        std = self.std(disc)

        # discrete
        cont = self.h_continuous(x)
        softmax_pdf = self.discrete_pdf(cont)

        # Scale output to [-action_bound, action_bound]
        mu = Lambda(lambda x: x * self.action_bound)(mu)

        return [mu, std, softmax_pdf]


# critic network
class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.v = Dense(1, activation='linear')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        v = self.v(x)
        return v


class HA2Cagent(object):

    def __init__(self, env):

        # hyperparameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        # environment
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        # networks
        self.actor = Actor(self.action_bound)
        self.critic = Critic()
        self.actor.build(input_shape=(None, self.state_dim))
        self.critic.build(input_shape=(None, self.state_dim))
        self.actor.summary()
        self.critic.summary()

        # optimizers
        self.actor_opt = adam_v2.Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = adam_v2.Adam(self.CRITIC_LEARNING_RATE)

        # save the results
        self.save_epi_reward = []

    # log of gaussian probability density function
    def log_gaussian_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def log_softmax_pdf(self, softmax_pdf, actions):

        idx = 0
        index = []
        for i in range(len(actions)):
            tmp = [idx, int(actions.numpy()[i])]
            idx += 1
            index.append(tmp)

        qgather = tf.gather_nd(softmax_pdf, index)

        return qgather

    def get_action(self, state):
        mu_a, std_a, pdf = self.actor(state)

        # unpack outer []
        mu_a = mu_a.numpy()[0]
        std_a = std_a.numpy()[0]
        pdf = pdf.numpy()[0]

        # clip stddev
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])

        # sample action
        action_continuous = np.random.normal(mu_a, std_a, size=self.action_dim)[0]
        action_discrete = np.random.choice(2, 1, p=pdf)[0]

        # clip continuous action
        action_continuous = np.clip(action_continuous, 0, self.action_bound)

        return [action_discrete, action_continuous]

    ## train the actor network
    def train_actor(self, states, actions, advantages):

        with tf.GradientTape() as tape:
            # policy pdf
            mu_a, std_a, pdf = self.actor(states, training=True)
            log_policy_pdf = self.log_gaussian_pdf(mu_a, std_a, actions[:, 1]) + self.log_softmax_pdf(pdf, actions[:, 0])

            # loss function and its gradients
            loss_policy = log_policy_pdf * advantages
            loss = tf.reduce_sum(-loss_policy)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

    def td_target(self, rewards, next_v_values, dones):
        td_targets = np.zeros(next_v_values.shape)
        for i in range(next_v_values.shape[0]):  # number of batch
            if dones[i]:
                td_targets[i] = rewards[i]
            else:
                td_targets[i] = rewards[i] + self.GAMMA * next_v_values[i]
        return td_targets

    def train_critic(self, states, td_targets):
        with tf.GradientTape() as tape:
            # get state value estimated by critic
            td_hat = self.critic(states, training=True)
            # get mse between estimated and target state value
            loss = tf.reduce_mean(tf.square(td_targets - td_hat))

        # get gradient
        grads = tape.gradient(loss, self.critic.trainable_variables)
        # optimize
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

    def load_weights(self, path):
        self.actor.load_weights(path + 'pendulum_actor.h5')
        self.critic.load_weights(path + 'pendulum_critic.h5')

    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)
        return unpack

    def train(self, max_episode_num):

        for ep in range(int(max_episode_num)):

            # batches. use mutable array for efficiency. use unpack() to convert back to immutable tf tensor.
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []

            time, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                # self.env.render()

                # first layer of actor(dense) requires 2 dim tensor at least. add a rank on state with []
                action = self.get_action(tf.convert_to_tensor([state], dtype=tf.float32))

                # reformat action suitable for env
                env_action = action[1] * (-1)**action[0]

                # observe
                next_state, reward, done, _ = self.env.step(action)

                # normalize reward to be within [0, 1]
                normalized_reward = (reward + 8) / 8

                # append to the batches(mutable arrays for efficiency)
                # batches can be converted back to immutable tf tensors using unpack()
                # [1 2 3].append([4 5 6]) returns [1 2 3 4 5 6] which eliminates borders between states, actions, etc.
                # use [] to address the problem
                batch_state.append([state])
                batch_action.append([action])
                batch_reward.append([normalized_reward])
                batch_next_state.append([next_state])
                batch_done.append([done])

                # check if batches are full
                if len(batch_state) < self.BATCH_SIZE:
                    # batch not full yet
                    state = next_state
                    episode_reward += reward
                    time += 1
                    continue

                # batches full. not continued.
                # unpack batches
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                rewards = self.unpack_batch(batch_reward)
                next_states = self.unpack_batch(batch_next_state)
                dones = self.unpack_batch(batch_done)

                # clear batches
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []

                # get next state value of each transition
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))

                # get td target of each transition
                td_targets = self.td_target(rewards, next_v_values.numpy(), dones)

                # train critic
                self.train_critic(tf.convert_to_tensor(states, dtype=tf.float32),
                                  tf.convert_to_tensor(td_targets, dtype=tf.float32))

                # compute baseline(current state value)
                baseline = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))

                # compute advantage
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))
                advantages = rewards + self.GAMMA * next_v_values - baseline

                # train actor
                self.train_actor(tf.convert_to_tensor(states, dtype=tf.float32),
                                 tf.convert_to_tensor(actions, dtype=tf.float32),
                                 tf.convert_to_tensor(advantages, dtype=tf.float32))

                # update state
                state = next_state
                episode_reward += reward
                time += 1

            # print progress
            print('Episode: ', ep + 1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)

            # save weight every 10 transitions
            if ep % 10 == 0:
                self.actor.save_weights("./save_weights/pendulum_actor.h5")
                self.critic.save_weights("./save_weights/pendulum_critic.h5")

        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
