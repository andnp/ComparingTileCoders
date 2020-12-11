import os
import sys
from typing import Any, Callable, Sequence
import numpy as np
sys.path.append(os.getcwd())

import PyExpUtils.utils.path as Path
import PyExpUtils.results.results as Results
from PyExpUtils.results.backends.backend import ResultList

def getBest(results: ResultList, reducer: Callable[[np.ndarray], Any]):
    results = map(lambda r: r.reducer(reducer), results)
    return Results.getBest(results, prefer='big')

def getCurveReducer(bestBy: str):
    if bestBy == 'auc':
        return np.mean

    if bestBy == 'end':
        return lambda m: np.mean(m[-int(m.shape[0] * .1):])

    raise NotImplementedError('Only now how to plot by "auc" or "end"')

def rename(alg: str, exp_path: str):
    return alg

def findExpPath(arr: Sequence[str], alg: str):
    for exp_path in arr:
        if f'{alg.lower()}.json' == Path.fileName(exp_path.lower()):
            return exp_path

    raise Exception(f'Expected to find exp_path for {alg}')
