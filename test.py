import gym
env = gym.make("RocketLander-v0")
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Action: {} Observations Size:{} score: {}".format(action,obs.shape,reward))
    if done:
        break