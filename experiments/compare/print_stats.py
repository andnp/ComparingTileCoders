import os
import sys
import numpy as np

sys.path.append(os.getcwd() + '/src')

from PyExpUtils.results.results import loadResults, whereParametersEqual, splitOverParameter
from experiment.tools import iterateDomains, parseCmdLineArgs
from experiment import ExperimentModel

def printStats(exp_paths, metric):
    print(f'-------------{metric}-------------')
    exp = ExperimentModel.load(exp_paths[0])
    results = loadResults(exp, f'{metric}.csv')

    results_dict = splitOverParameter(results, 'tile-coder')

    # two new streams over results
    mine = results_dict['mine']
    tiles3 = results_dict['tiles3']

    # figure out what values of "tiles" we swept
    tiles = splitOverParameter(mine, 'tiles').keys()
    # figure out what values of "tilings" we swept
    tilings = splitOverParameter(mine, 'tilings').keys()

    for num_tiles in tiles:
        for num_tilings in tilings:
            mine_results = list(whereParametersEqual(mine, { 'tiles': num_tiles, 'tilings': num_tilings }))
            tiles3_results = list(whereParametersEqual(tiles3, { 'tiles': num_tiles, 'tilings': num_tilings }))

            mine_means = []
            tiles3_means = []

            # loop over each value of alpha
            # this way we just get 3x as many samples of timing
            for i in range(len(mine_results)):
                mine_mean = mine_results[i].mean()[0]
                tiles3_mean = tiles3_results[i].mean()[0]

                mine_means.append(mine_mean)
                tiles3_means.append(tiles3_mean)

            mine_mean = np.mean(mine_means)
            tiles3_mean = np.mean(tiles3_means)

            # TODO: this is covering up a bug in results. Rerun results
            if metric == 'feature_utilization':
                mine_mean = mine_mean / (num_tilings * num_tiles**2)
                tiles3_mean = tiles3_mean / (num_tilings * num_tiles**2)

            print(f'({num_tiles}, {num_tilings}) -- {mine_mean}, {tiles3_mean}')

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    for domain in iterateDomains(path):
        exp_paths = domain.exp_paths
        save_path = domain.save_path

        for metric in ['time', 'feature_utilization']:
            printStats(exp_paths, metric)
