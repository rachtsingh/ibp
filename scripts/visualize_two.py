from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import sys

# relative path import hack
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from src.utils import visualize_A

sns.set()

def main():
    a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
    print(len(a.shape))
    if len(a.shape) == 1:
        # ELBO plot
        ax = plt.gca()
        ax.plot(a, color='green')
        ax.plot(b, color='red')
        plt.show()
    elif len(a.shape) == 3:
        visualize_A(a[-1])
        visualize_A(b[-1])

if __name__ == '__main__':
    main()