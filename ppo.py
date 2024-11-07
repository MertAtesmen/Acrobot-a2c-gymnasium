import tensorflow as tf
from tensorflow.keras import layers, optimizers
import numpy as np
import gymnasium as gym

env = gym.make('Acrobot-v1')
action_space = env.action_space.n

class Model(tf.keras.Model):
    def __init__(self, action_space):
        super().__init__()
        # Actor network
        self.actor_h1 = layers.Dense(128, activation='relu', name='act1')
        self.actor_h2 = layers.Dense(64, activation='relu', name='act2')
        self.actor_logits = layers.Dense(action_space, activation=None, name='actor_logits')  # No activation
        # Critic network
        self.val_h1 = layers.Dense(128, activation='relu', name='val1')
        self.val_h2 = layers.Dense(64, activation='relu', name='val2')
        self.val_logit = layers.Dense(1, activation=None, name='val_logit')  # No activation

    def call(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        # Actor forward pass
        x_act = self.actor_h1(x)
        x_act = self.actor_h2(x_act)
        act_logits = self.actor_logits(x_act)
        # Critic forward pass
        x_val = self.val_h1(x)
        x_val = self.val_h2(x_val)
        val = self.val_logit(x_val)
        return act_logits, val

class Agent:
    def __init__(self, model):
        self.model = model
        self.num_rollouts = 500 
        self.val_weight = 0.5
        self.model_entropy = 1e-3
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.epsilon = 0.2  # Clipping parameter for PPO
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)

    def train(self, env):
        rollout_scores = []
        for rollout in range(self.num_rollouts):
            obs = env.reset()[0].reshape((1, -1))
            observations = []
            rollout_history = []
            rollout_score = 0
            done = False
            while not done:
                observation = obs.copy()
                observations.append(observation)
                act_logits, val = self.model(obs)
                act_logits = tf.convert_to_tensor(act_logits)
                val = tf.convert_to_tensor(val)
                # Sample action from the policy
                action = tf.squeeze(tf.random.categorical(act_logits, 1), axis=-1).numpy()
                action = np.squeeze(action, axis=-1).item()
                val = np.squeeze(val, axis=-1).item()
                # Compute log probability of the action
                log_probs = tf.nn.log_softmax(act_logits)
                action_log_prob = log_probs[0, action].numpy()
                # Step the environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                obs = obs.reshape((1, -1))
                if done:
                    reward -= 10
                # Store experience
                rollout_history.append([reward, val, action, action_log_prob])
                rollout_score += reward
            rollout_scores.append(rollout_score)
            # Convert experience to arrays
            rollout_history = np.array(rollout_history)
            rewards = rollout_history[:, 0]
            values = rollout_history[:, 1]
            actions = rollout_history[:, 2].astype(np.int32)
            old_action_log_probs = rollout_history[:, 3]
            returns, advantages = self._discount_rewards(rewards, values)
            observations = np.array(observations).reshape(-1, env.observation_space.shape[0])
            # Update the policy and value function
            with tf.GradientTape() as tape:
                act_logits, vals = self.model(observations)
                vals = tf.squeeze(vals, axis=-1)
                # Compute new log probabilities
                new_log_probs = tf.nn.log_softmax(act_logits)
                indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
                new_action_log_probs = tf.gather_nd(new_log_probs, indices)
                # Ratio for PPO
                ratio = tf.exp(new_action_log_probs - old_action_log_probs)
                # Clipped surrogate objective
                advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                # Value function loss
                returns = tf.convert_to_tensor(returns, dtype=tf.float32)
                val_loss = tf.reduce_mean(tf.square(returns - vals))
                # Entropy loss for exploration
                entropy = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(act_logits) * tf.nn.log_softmax(act_logits), axis=1))
                # Total loss
                total_loss = policy_loss + self.val_weight * val_loss - self.model_entropy * entropy
            # Update model parameters
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            print(f"Episode {rollout}: Score {np.mean(rollout_scores[-100:])}")

    def test(self, env):
        sample = np.random.rand(env.observation_space.shape[0])
        sample = np.reshape(sample, (1, -1))
        self.model(sample)
        self.model.load_weights('saved/acrobot/ppo_weights.h5')
        obs, done, ep_reward = env.reset()[0], False, 0
        while not done:
            act_logits, _ = self.model(obs[None, :])
            action = tf.squeeze(tf.random.categorical(act_logits, 1), axis=-1)
            action = np.squeeze(action, axis=-1)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            env.render()
        return ep_reward

    def _discount_rewards(self, rewards, values):
        returns = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            returns[t] = running_add
        advantages = returns - values
        return returns, advantages

model = Model(env.action_space.n)
agent = Agent(model)
agent.train(env)
model.save_weights('saved/acrobot/ppo.weights.h5')
