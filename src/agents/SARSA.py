import numpy as np
from typing import Dict
from numba import njit
from utils.policies import buildEGreedyPolicy

@njit(cache=True)
def _update(w, x, a, xp, ap, r, gamma, alpha):
    q_a = w[a].dot(x)
    qp_ap = w[ap].dot(xp)

    delta = r + gamma * qp_ap - q_a

    w[a] = w[a] + alpha * delta * x

class SARSA:
    def __init__(self, features: int, actions: int, params: Dict, seed: int):
        self.features = features
        self.actions = actions
        self.params = params

        self.random = np.random.RandomState(seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        # create initial weights
        self.w = np.zeros((actions, features))

        # use the policy utility class to keep track of policy probabilities,
        # sample actions, and maintain rng state
        self.policy = buildEGreedyPolicy(self.random, self.epsilon, lambda x: self.w.dot(x))

    def selectAction(self, x):
        return self.policy.selectAction(x)

    def update(self, x, a, xp, r, gamma):
        ap = self.selectAction(xp)

        # defer the bulk of the update to a fast compiled function
        _update(self.w, x, a, xp, ap, r, gamma, self.alpha)
        return ap
