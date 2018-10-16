import numpy as np 
from numpy.random import poisson, beta
import matplotlib.pyplot as plt
from scipy import stats

FILE_PATH = '~/Documents/class/current/cs282/1/'

def left_order_form(m):
    twos = np.ones(m.shape[0]) * 2.0
    twos[0] = 1.0
    powers = np.cumprod(twos)[::-1]
    values = np.dot(powers, m)
    idx = values.argsort()[::-1]
    return np.take(m, idx, axis=1)

# basically make it so that the columns are in the order that the IBP has
def order_by_column_counts(m):
    sums = np.sum(m, axis=0)
    idx = sums.argsort()[::-1]
    return np.take(m, idx, axis=1)

# order by the maximum feature that is nonzero (so that it's in IBP form)
def order_by_max_dish(m):
    maximums = []
    for row in range(m.shape[0]):
        maximums.append(0)
        for i in range(m.shape[1]):
            if m[row][i]:
                maximums[row] = i
    new_idx = np.array(maximums).argsort()
    return np.take(m, new_idx, axis=0)

# one parameter version
def weak_limit_approximation(n=20, alpha=5.0, size=100):
    ret = np.zeros((n, size))
    mus = beta(float(alpha)/size, 1 - 1.0/size, (size))
    selected = np.random.random((n, size)) < np.tile(mus, (n, 1)).reshape((n, size))
    ret[selected] = 1.0
    return left_order_form(ret)

def alternative_weak_limit(n=20, alpha=5.0, size=100):
    ret = np.zeros((n, size))
    mus = beta(float(alpha)/size, 1, (size))
    selected = np.random.random((n, size)) < np.tile(mus, (n, 1)).reshape((n, size))
    ret[selected] = 1.0
    return left_order_form(ret)

def indian_buffet_process(n=20, alpha=5.0, size=100):
    ret = np.zeros((n, size))
    total_dishes_sampled = 0
    for i in range(n):
        selected = np.random.random(total_dishes_sampled) < np.sum(ret[:, :total_dishes_sampled], axis=0)/float(i + 1)
        ret[i][:total_dishes_sampled][selected] = 1.0
        new_dishes = poisson(alpha/(i + 1))
        if total_dishes_sampled + new_dishes >= size:
            new_dishes = size - total_dishes_sampled
        ret[i][total_dishes_sampled:total_dishes_sampled + new_dishes] = 1.0
        total_dishes_sampled += new_dishes
    return left_order_form(ret)

def stick_breaking_construction(n=20, alpha=5.0, size=100):
    ret = np.zeros((n, size))
    mus = np.cumprod(beta(alpha, 1.0, (size)))
    selected = np.random.random((n, size)) < np.tile(mus, (n, 1)).reshape((n, size))
    ret[selected] = 1.0
    return left_order_form(ret)

# plot the general matrixes produced
def generate_plots():
    methods = {
        'Weak Limit': weak_limit_approximation,
        'Indian Buffet': indian_buffet_process, 
        'Stick Breaking': stick_breaking_construction,
    }

    f, axes = plt.subplots(5, 3)
    for j, name in enumerate(methods):
        axes[0][j].set_title(name)
        for i in range(5):
            matrix = methods[name](50, 10.0, 100)
            axes[i][j].imshow(matrix, cmap='Greys', interpolation='nearest')
            # sns.heatmap(matrix, ax=axes[i][j], cmap='Greys', cbar=False)
    f.set_size_inches(8, 10)
    f.savefig('images/samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_feature_counts():
    methods = {
        'Weak Limit': weak_limit_approximation,
        'Indian Buffet': indian_buffet_process, 
        'Stick Breaking': stick_breaking_construction,
    }

    ns = [100, 500, 1000]
    sizes = [500, 1250, 2500]

    plot_size = 15 # should be arg
    number_iterations = 50

    f, axes = plt.subplots(3, 3)
    for j, name in enumerate(methods):
        axes[0][j].set_title(name)
        axes[2][j].set_xlabel('# of features')
        for i in range(3):
            axes[i][j].set_ylabel("%d samples, %d features" % (ns[i], sizes[i]), size='small')
            axes[i][j].tick_params(axis='x', labelsize=6)
            axes[i][j].tick_params(axis='y', labelsize=6)
            counts = np.zeros(plot_size)
            for r in range(number_iterations):
                matrix = methods[name](ns[i], 3.0, sizes[i]).astype('int64')
                tcounts = np.bincount(np.sum(matrix, axis=1))
                tcounts.resize(plot_size)
                counts += tcounts
            counts /= float(number_iterations)
            axes[i][j].plot(np.arange(plot_size), counts / float(np.sum(counts)), color='red')
            axes[i][j].plot(np.arange(plot_size), stats.poisson(3.0).pmf(np.arange(plot_size)))
            axes[i][j].set_ylim([0, 0.5])
    f.set_size_inches(7, 4)
    f.savefig('images/feature_counts.png', dpi=300)
    plt.show()

def generate_features_per_observation():
    # plt.rc('text', usetex=True)

    def ibp_numbers(n=100, iterations=10):
        features = np.zeros(n)
        xes = np.arange(1, n + 1)
        for r in range(iterations):
            tfeatures = np.zeros(n)
            matrix = indian_buffet_process(n, 10.0, 1000).astype('int64')
            for i in range(n):
                tfeatures[i] = max(tfeatures[i-1], np.max(matrix[i] * np.arange(1, 1001)))
            features += tfeatures
        features /= float(iterations)
        return features

    def weak_limit_numbers(n=100, iterations=10):
        features = np.zeros(n)
        xes = np.arange(1, n + 1)
        for r in range(iterations):
            tfeatures = np.zeros(n)
            matrix = weak_limit_approximation(n, 10.0, 10000).astype('int64')
            for i in range(n):
                tfeatures[i] = max(tfeatures[i-1], np.max(matrix[i] * np.arange(1, 10001)))
            features += tfeatures
        features /= float(iterations)
        return features

    def stick_breaking_numbers(n=100, iterations=10):
        features = np.zeros(n)
        xes = np.arange(1, n + 1)
        for r in range(iterations):
            tfeatures = np.zeros(n)
            matrix = stick_breaking_construction(n, 10.0, 10000).astype('int64')
            for i in range(n):
                tfeatures[i] = max(tfeatures[i-1], np.max(matrix[i] * np.arange(1, 10001)))
            features += tfeatures
        features /= float(iterations)
        return features

    methods = {
        'Weak Limit': weak_limit_numbers,
        'Indian Buffet': ibp_numbers, 
        'Stick Breaking': stick_breaking_numbers,
    }

    N = 100

    sums = np.cumsum(np.ones(N) / np.arange(1, N + 1))

    f, axes = plt.subplots(1, 3)
    for j, name in enumerate(methods):
        axes[j].set_title(name)
        axes[j].set_ylabel('# of features seen')
        axes[j].set_xlabel('# of observations')
        xes = np.arange(1, N + 1)
        features = methods[name](N, 250)
        axes[j].plot(np.arange(1, N + 1), features, color='blue', label='sample')
        axes[j].plot(xes, 10.0 * (np.log(xes)), color='green', label='lower bound')
        axes[j].plot(xes, 10.0 * (np.log(xes) + 1), color='green', label='upper bound')
        axes[j].plot(xes, 10.0 * sums, color='red', label='harmonic sum')
        if j == 2:
            handles, labels = axes[j].get_legend_handles_labels()
            axes[j].legend(handles, labels, bbox_to_anchor=(1, 0.2), prop={'size':6})
    f.set_size_inches(8, 8)
    f.savefig('images/features_per_observation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_k_better():
    def weak_limit_numbers(n=100, iterations=10, K=100, alpha=10.0):
        last = 0.0
        for r in range(iterations):
            matrix = weak_limit_approximation(n, alpha, K).astype('int64')
            last += len(np.nonzero(np.sum(matrix, axis=0))[0])
        last /= float(iterations)
        return last

    def stick_breaking_numbers(n=100, iterations=10, K=100, alpha=10.0):
        last = 0.0
        for r in range(iterations):
            matrix = stick_breaking_construction(n, alpha, K).astype('int64')
            last += len(np.nonzero(np.sum(matrix, axis=0))[0])
        last /= float(iterations)
        return last

    N = 100
    sums = np.cumsum(np.ones(N) / np.arange(1, N + 1))

    Ks = sorted([25, 50, 75, 100, 150, 200, 250, 1000, 1200, 1400, 1600, 1800, 2000, 2500] + list(np.arange(5, 25) * 40))
    alphas = [1.0, 5.0, 10.0, 25.0, 50.0]
    f, axes = plt.subplots(5, 1)
    for i, alpha in enumerate(alphas):
        lasts = np.array([stick_breaking_numbers(N, 100, k, alpha) for k in Ks])
        axes[i].plot(np.array(Ks), (lasts - (alpha * sums[-1]))/alpha, color='red')
        axes[i].plot(np.array(Ks), np.zeros(len(Ks)), color='green')
        axes[i].set_title("a = " + str(alpha))
        axes[i].set_ylabel('error')
        if i == len(alphas) - 1:
            axes[i].set_xlabel('K')
        xes = np.arange(1, N + 1)

    f.set_size_inches(8, 8)
    plt.show()

def sample_Z_restaurant(n=100, alpha=2.0, K=1000):
    return indian_buffet_process(n, alpha, K), None

def main():
    generate_features_per_observation()

if __name__ == '__main__':
    main()