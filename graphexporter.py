import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.signal
import matplotlib

plt.style.use('dark_background')

# with open('csv/run_PPO_summary-tag-Info_cumulative_reward.csv', newline='') as csvfile:
with open('csv/run_PPO_summary-tag-Info_episode_length.csv', newline='') as csvfile:
# with open('csv/run_PPO_summary-tag-Info_value_loss.csv', newline='') as csvfile:
# with open('csv/run_PPO_summary-tag-Info_policy_loss.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    csvlist = np.array(list(csvreader))
    steplist = csvlist[1:, 1].astype(np.int32)
    datalist = csvlist[1:, 2].astype(np.float) * 6

datalist = scipy.signal.medfilt(datalist, kernel_size=15)
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], animated=True)

plt.xlabel('timesteps')
plt.ylabel('frames')  # <-----
plt.title('episode length')  # <-----


def init():
    ax.set_ylim(0, 700)  # <-----
    ax.set_xlim(0, 15e6)
    return ln,


def update(i):
    j = int(i / 10)
    delta = (i % 10) / 10
    print(i, j)

    xdata.append(steplist[j] + delta * (steplist[j + 1] - steplist[j]))
    ydata.append(datalist[j] + delta * (datalist[j + 1] - datalist[j]))
    ln.set_data(xdata, ydata)
    return ln,


ani = FuncAnimation(fig, update, repeat=False, frames=(len(steplist) - 1) * 10,
                    init_func=init, blit=True, interval=1 / 60 * 1000)

ani.save('episode_length.mp4', dpi=120, writer='ffmpeg')  # <-----
# plt.show()
