#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of bandits using different methods or strategies to estimate the desired value
"""

# Adapted from: https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python

import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, true_mean, initial_mean=0):
        self.true_mean = true_mean
        self.predicted_mean = initial_mean
        self.N = 0  # 0 in epsilon, 1 in optimistic?

    def pull(self):
        return np.random.randn() + self.true_mean

    def update(self, x):
        self.N += 1
        self.predicted_mean = (1 - 1.0/self.N)*self.predicted_mean + (1.0/self.N)*x


class BayesianBandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        # parameters for mu - prior is N(0,1)
        self.predicted_mean = 0
        self.lambda_ = 1
        self.sum_x = 0  # for convenience
        self.tau = 1

    def pull(self):
        return np.random.randn() + self.true_mean

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.predicted_mean

    def update(self, x):
        # assume tau is 1
        self.lambda_ += 1
        self.sum_x += x
        self.predicted_mean = self.tau*self.sum_x / self.lambda_


def ucb(mean, n, nj):
    return mean + np.sqrt(2*np.log(n) / (nj + 1e-2))


def epsilon_greedy(v, epsilon=0.01, decaying=False):
    p = np.random.random()
    if p < epsilon:
        j = np.random.choice(len(v))
    else:
        j = np.argmax([vv.mean for vv in v])

    return j


def run_experiment(m_values, n, eps=0.01, decaying_epsilon=False, optimistic=False, upper_limit=10,
                   use_ucb=True, bayesian=False):

    # --- Choose class to use based on the method ---
    if bayesian is True:
        bandits = [BayesianBandit(i) for i in m_values]
    elif optimistic is True:
        bandits = [Bandit(i, upper_limit) for i in m_values]
    else:
        bandits = [Bandit(i) for i in m_values]

    data = np.empty(n)

    for i in range(n):

        # --- Explore/exploit ---

        if bayesian is True:
            # bayesian/thompson sampling
            j = np.argmax([b.sample() for b in bandits])

        elif optimistic is True:
            # optimistic initial values
            if use_ucb is True:
                j = np.argmax([ucb(b.predicted_mean, i+1, b.N) for b in bandits])
            else:
                j = np.argmax([b.predicted_mean for b in bandits])
        else:
            # epsilon greedy
            p = np.random.random()

            if decaying_epsilon is True:
                # override epsilon with the decaying form
                eps = 1.0 / (i+1)

            if p < eps:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.predicted_mean for b in bandits])

        x = bandits[j].pull()
        bandits[j].update(x)

        # for the plot
        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(n) + 1)

    # --- Print information ---
    print("---------------------------------")
    if bayesian is True:
        print("> Experiment: %i bandits using bayesian sampling" % len(bandits))
    elif optimistic is True:
        if use_ucb is True:
            print("> Experiment: %i bandits using UCB1" % len(bandits))
        else:
            print("> Experiment: %i bandits using optimistic values" % len(bandits))
    else:
        if decaying_epsilon is True:
            print("> Experiment: %i bandits using decaying epsilon" % len(bandits))
        else:
            print("> Experiment: %i bandits using epsilon=%s" % (len(bandits), str(eps)))

    for i in range(len(bandits)):
        b = bandits[i]
        print("Bandit %i:" % i)
        print("True mean: %s" % str(b.true_mean))
        print("Predicted mean: %s" % str(b.predicted_mean))

    return cumulative_average

# --- Available methods ---
methods = {'epsilon_greedy': lambda args: run_experiment(m_values=args['m'], n=args['n'], optimistic=False,
                                                         eps=args['epsilon'], decaying_epsilon=False),
           'decaying_epsilon_greedy': lambda args: run_experiment(m_values=args['m'], n=args['n'],
                                                                  optimistic=False, decaying_epsilon=True),
           'optimistic_initial_values': lambda args: run_experiment(m_values=args['m'], n=args['n'],
                                                                    optimistic=True, upper_limit=args['upper_limit'],
                                                                    use_ucb=False),
           'UCB1': lambda args: run_experiment(m_values=args['m'], n=args['n'], optimistic=True, use_ucb=True,
                                               upper_limit=args['upper_limit']),
           'bayesian': lambda args: run_experiment(m_values=args['m'], n=args['n'], bayesian=True)
           }

if __name__ == '__main__':

    # --- Configuration ---
    # parameters of experiment
    epsilons = [0.1, 0.05, 0.01]
    N = 100000
    true_m = [1.0, 2.0, 3.0]
    upper_limit_value = 10

    optimistic_values = True
    use_ucb1 = True

    # methods to compare
    methods_comp = methods.keys()

    # parameters of plot
    scale_log = True

    # annotations
    plt.title('Bandits performance')
    plt.xlabel('Samples')
    plt.ylabel('Mean')

    # limits of plot
    #plt.xlim([1, N])
    plt.ylim([min(true_m) - 1.0, max(true_m) + 0.5])

    # --- Compute and plot ---
    # plot true means as horizontal lines
    for t_m in true_m:
        plt.plot(np.ones(N)*t_m)

    # compute estimation and plot

    for meth in methods_comp:

        args = {'m': true_m, 'n': N, 'upper_limit': upper_limit_value}

        if meth == 'epsilon_greedy':
            for ep in epsilons:
                args['epsilon'] = ep
                c = methods[meth](args)
                # plot with legend of used epsilon
                plt.plot(c, label='eps = %s' % str(ep))
        else:
            c = methods[meth](args)
            plt.plot(c, label=meth)

    # labels
    plt.legend(loc='lower right')

    # scale, to see the differences better
    if scale_log is True:
        plt.xscale('log')

    # plot!
    plt.show()
