import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import numpy as np
import gym

env = gym.make('Acrobot-v1')
action_space = env.action_space.n


class Model(tf.keras.Model):
    def __init__(self, action_space):
        super().__init__()
        self.actor_h1 = layers.Dense(128, activation='relu')
        self.actor_h2 = layers.Dense(64, activation='relu')
        self.actor_logits = layers.Dense(action_space, activation='softmax', name="actor_logits")
        self.val_h1 = layers.Dense(128, activation='relu')
        self.val_h2 = layers.Dense(64, activation='relu')
        self.val_logit = layers.Dense(1, name='val_logit')

    def call(self, x):

        x = tf.convert_to_tensor(x, dtype=tf.float32)

        x_act = self.actor_h1(x)
        x_act = self.actor_h2(x_act)
        x_act = self.actor_logits(x_act)

        x_val = self.val_h1(x)
        x_val = self.val_h2(x)
        x_val = self.val_logit(x_val)

        return x_act, x_val


class Agent:

    def __init__(self, model):
        self.model = model
        self.num_rollouts = 1000
        self.batch_size = 100
        self.val_weight = .4
        self.model_entropy = 1e-3
        self.learning_rate = 1e-3
        self.gamma = .95
        self.optimizer=optimizers.Adam(learning_rate = self.learning_rate)

    def train(self,env):
        rollout_scores = []
        for rollout in range(self.num_rollouts):
            obs = env.reset().reshape((1,-1))
            observations = []
            rollout_history = []
            rollout_score = 0
            done = False
            render = False
            if rollout % 100 == 0:
                render = True
            while not done:
                observation = obs.copy()
                observations.append(observation)
                act_logits, val_logits = self.model.predict(obs)
                action = tf.squeeze(tf.random.categorical(act_logits, 1), axis=-1)
                action = np.squeeze(action, axis=-1)
                val = np.squeeze(val_logits, axis=-1)
                obs, reward, done, _ = env.step(action)
                obs = obs.reshape((1,-1))
                if done:
                    reward-=10
                rollout_history.append([reward, val, action])
                rollout_score += reward
                if render:
                    env.render()
                rollout_scores.append(rollout_score)
                if done:
                    rollout_history = np.array(rollout_history)
                    returns, advantages = self._discount_rewards(rollout_history[:,0], rollout_history[:,1])
                    observations = np.reshape(observations, (len(observations),6))
                    with tf.GradientTape() as tape:
                        logits, vals = self.model(observations)
                        actions = tf.convert_to_tensor(rollout_history[:,2], dtype=tf.int32)
                        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
                        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
                        logits_loss = self._actor_loss(advantages,actions,logits)
                        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
                        vals = tf.convert_to_tensor(vals, dtype=tf.float32)
                        val_loss = self._critic_loss(returns,vals)
                    gradients = tape.gradient([logits_loss,val_loss],model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients,model.trainable_variables))

            print(f"Episode  {rollout}:  Score  {np.mean(rollout_scores[-100:])}")

    
    def _discount_rewards(self, rewards, values):
        returns = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            running_add = running_add * self.gamma + rewards[t]
            returns[t] = running_add
        advantages = returns - values
        return returns, advantages

    @tf.function
    def _actor_loss(self, advantages, actions, logits):
        weighted_sparse_ce = losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        entropy_loss = losses.categorical_crossentropy(logits,logits,from_logits=True)
        return policy_loss - self.model_entropy*entropy_loss

    @tf.function
    def _critic_loss(self, returns, values):
        return self.val_weight*losses.mean_squared_error(returns,values)


model = Model(env.action_space.n)
agent = Agent(model)
agent.train(env)
