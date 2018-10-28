import numpy as np
import sys
import matplotlib
import pdb
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.verbose.set_level('helpful')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist=''), bitrate=1800)

if len(sys.argv) == 1:
    path = 'learned_features.npy'
else:
    path = sys.argv[1]

data = np.load(path)
features = data
a, b = features.min(), features.max()

loops = []

fig, axes = plt.subplots(3, 2)
for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        loops.append(ax.imshow(np.zeros((6, 6)), cmap='Greys', vmin=a, vmax=b, interpolation=None, animated=True))

epoch = 0

def updatefig(frame, *fargs):
    global epoch
    global features
    global loops
    plt.title('{}'.format(epoch))
    epoch += 1
    for i, loop in enumerate(loops):
        loop.set_array(frame[i].reshape((6, 6)))
    return loops


ani = animation.FuncAnimation(fig, updatefig, features, interval=5, blit=True)
ani.save('advi_exact.mp4', writer=writer)