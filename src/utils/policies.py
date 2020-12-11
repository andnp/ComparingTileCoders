import numpy as np
from numba import njit
from typing import Any, Callable, Sequence
from PyExpUtils.utils.types import NpList
from PyExpUtils.utils.random import sample
from PyExpUtils.utils.arrays import argsmax

class Policy:
    def __init__(self, probs: Callable[[Any], NpList], rng = np.random):
        self.probs = probs
        self.random = rng

    def selectAction(self, s: Any):
        action_probabilities = self.probs(s)
        return sample(action_probabilities, rng=self.random)

    def ratio(self, other: Any, s: Any, a: int) -> float:
        probs = self.probs(s)
        return probs[a] / other.probs(s)[a]

def fromStateArray(probs: Sequence[NpList]):
    return Policy(lambda s: probs[s])

def fromActionArray(probs: NpList):
    return Policy(lambda s: probs)

@njit(cache=True)
def egreedy_probabilities(qs: np.ndarray, epsilon: float):
    actions = len(qs)

    # compute the greedy policy
    max_acts = argsmax(qs)
    pi = np.zeros(actions)
    for a in max_acts:
        pi[a] = 1. / len(max_acts)

    # compute a uniform random policy
    uniform = np.ones(actions) / actions

    # epsilon greedy is a mixture of greedy + uniform random
    return (1. - epsilon) * pi + epsilon * uniform

def buildEGreedyPolicy(rng, epsilon: float, getValues: Callable[[Any], np.ndarray]):
    return Policy(lambda state: egreedy_probabilities(getValues(state), epsilon), rng)
