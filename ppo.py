import os
import shutil
import threading

from ppo.models import *
from ppo.trainer import Trainer
from agents import GymEnvironment

# ## Proximal Policy Optimization (PPO)
# Contains an implementation of PPO as described [here](https://arxiv.org/abs/1707.06347).

# General parameters
# keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
keep_checkpoints = 5
# load                     Whether to load the model or randomly initialize [default: False].
load_model = False
# max-steps=<n>            Maximum number of steps to run environment [default: 1e6].
max_steps = 6e6
# run-path=<path>          The sub-directory name for model and summary statistics.
summary_path = '/home/sven/Desktop/PPO_summary'
model_path = './models'
# save-freq=<n>            Frequency at which to save model [default: 50000].
save_freq = 500000
# summary-freq=<n>         Frequency at which to save training statistics [default: 10000].
summary_freq = 10000
# train                    Whether to train model, or only run inference [default: False].
train_model = True
# render environment to display progress
render = True

# Algorithm parameters
# batch-size=<n>           How many experiences per gradient descent update step [default: 64].
batch_size = 128
# beta=<n>                 Strength of entropy regularization [default: 2.5e-3].
beta = 2.5e-3
# buffer-size=<n>          How large the experience buffer should be before gradient descent [default: 2048].
buffer_size = batch_size * 32
# epsilon=<n>              Acceptable threshold around ratio of old and new policy probabilities [default: 0.2].
epsilon = 0.2
# gamma=<n>                Reward discount rate [default: 0.99].
gamma = 0.99
# hidden-units=<n>         Number of units in hidden layer [default: 64].
hidden_units = 128
# lambd=<n>                Lambda parameter for GAE [default: 0.95].
lambd = 0.95
# learning-rate=<rate>     Model learning rate [default: 3e-4].
learning_rate = 3e-4
# normalize                Whether to normalize the state input using running statistics [default: False].
normalize = False
# num-epoch=<n>            Number of gradient descent steps per batch of experiences [default: 5].
num_epoch = 5
# num-layers=<n>           Number of hidden layers between state/observation and outputs [default: 2].
num_layers = 1
# time-horizon=<n>         How many steps to collect per agent before adding to buffer [default: 2048].
time_horizon = 1024

env_name = 'RocketLander-v0'
env = GymEnvironment(env_name=env_name, log_path="./PPO_log")

print(str(env))
brain_name = env.external_brain_names[0]

tf.reset_default_graph()

ppo_model = create_agent_model(env, lr=learning_rate,
                               h_size=hidden_units, epsilon=epsilon,
                               beta=beta, max_step=max_steps,
                               normalize=normalize, num_layers=num_layers)

is_continuous = env.brains[brain_name].action_space_type == "continuous"
use_observations = False
use_states = True

if not load_model:
    shutil.rmtree(summary_path, ignore_errors=True)

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

tf.set_random_seed(np.random.randint(1024))
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=keep_checkpoints)


# monitoring process to view progress
def watch(sess_):
    with sess_.as_default():
        env_monitor = GymEnvironment(env_name=env_name, log_path=None, render=True)
        info_ = env_monitor.reset()[brain_name]
        trainer_monitor = Trainer(ppo_model, sess_, info_, is_continuous, use_observations, use_states, False)
        steps_ = sess_.run([ppo_model.global_step])

        print("Starting watcher.")
        while True:
            done = False
            info_ = env_monitor.reset()[brain_name]
            while not done:
                info_ = trainer_monitor.take_action(info_, env_monitor, brain_name, steps_, normalize, stochastic=False)
                done = info_.local_done[0]


with tf.Session() as sess:
    # Instantiate model parameters
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt is None:
            print('The model {0} could not be found. Make sure you specified the right --run-path'.format(model_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    steps, last_reward = sess.run([ppo_model.global_step, ppo_model.last_reward])
    summary_writer = tf.summary.FileWriter(summary_path)
    info = env.reset()[brain_name]
    trainer = Trainer(ppo_model, sess, info, is_continuous, use_observations, use_states, train_model)
    episode_number = 0

    watcher_started = False

    while steps <= max_steps or not train_model:
        if env.global_done:
            episode_number += 1
            info = env.reset()[brain_name]
            trainer.reset_buffers(info, total=True)
        # Decide and take an action
        info = trainer.take_action(info, env, brain_name, steps, normalize, stochastic=True)
        trainer.process_experiences(info, time_horizon, gamma, lambd)
        if len(trainer.training_buffer['actions']) > buffer_size and train_model:
            # Perform gradient descent with experience buffer
            trainer.update_model(batch_size, num_epoch)
        if steps % summary_freq == 0 and steps != 0 and train_model:
            # Write training statistics to tensorboard.
            trainer.write_summary(summary_writer, steps, episode_number)
        if steps % save_freq == 0 and steps != 0 and train_model:
            # Save Tensorflow model
            save_model(sess, model_path=model_path, steps=steps, saver=saver)
        if train_model:
            steps += 1
            sess.run(ppo_model.increment_step)
            if len(trainer.stats['cumulative_reward']) > 0:
                mean_reward = np.mean(trainer.stats['cumulative_reward'])
                sess.run(ppo_model.update_reward, feed_dict={ppo_model.new_reward: mean_reward})
                last_reward = sess.run(ppo_model.last_reward)
        if not watcher_started and render:
            watcher = threading.Thread(target=watch, args=(tf.get_default_session(),))
            watcher.start()
            watcher_started = True
    # Final save Tensorflow model
    if steps != 0 and train_model:
        save_model(sess, model_path=model_path, steps=steps, saver=saver)
env.close()
export_graph(model_path, env_name)
