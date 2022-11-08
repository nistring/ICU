import numpy as np
import matplotlib.pyplot as plt


def compute_dist_acc_wrapper(pred, gt, max_dist=10, num=100):
    '''
    pred: (N, K, 3)
    gt: (N, K, 3)

    return dist: (K, )
    return acc: (K, num)
    '''
    assert(pred.shape == gt.shape)
    assert(len(pred.shape) == 3)

    dist = np.linspace(0, max_dist, num)
    return dist, compute_dist_acc(pred, gt, dist)


def compute_dist_acc(pred, gt, dist):
    '''
    pred: (N, K, 3)
    gt: (N, K, 3)
    dist: (M, )

    return acc: (K, M)
    '''
    assert(pred.shape == gt.shape)
    assert(len(pred.shape) == 3)

    N, K = pred.shape[0], pred.shape[1]
    err_dist = np.sqrt(np.sum((pred - gt)**2, axis=2))  # (N, K)

    acc = np.zeros((K, dist.shape[0]))

    for i, d in enumerate(dist):
        acc_d = (err_dist < d).sum(axis=0) / N
        acc[:,i] = acc_d

    return acc
        
    
def compute_mean_err(pred, gt):
    '''
    pred: (N, K, 3)
    gt: (N, K, 3)

    mean_err: (K,)
    '''
    N, K = pred.shape[0], pred.shape[1]
    err_dist = np.sqrt(np.sum((pred - gt)**2, axis=2))  # (N, K)
    avg_precision = np.sum(err_dist <= 0.1, axis=0) / N
    return np.mean(err_dist, axis=0), avg_precision


def plot_acc(ax, dist, acc, name):
    '''
    acc: (K, num)
    dist: (K, )
    name: (K, )
    '''
    assert(acc.shape[0] == len(name))

    for i in range(len(name)):
        ax.plot(dist, acc[i], label=name[i])

    ax.legend()

    ax.set_xlabel('Maximum allowed distance to GT (m)')
    ax.set_ylabel('Fraction of samples within distance')


def plot_mean_err(ax, mean_err, name):
    '''
    mean_err: (K, )
    name: (K, )
    '''
    ax.bar(name, mean_err)
    ax.set_ylabel('Mean error distance to GT (m)')
