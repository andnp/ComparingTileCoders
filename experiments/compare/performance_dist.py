import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.getcwd() + '/src')

from scipy.stats.kde import gaussian_kde
from PyExpUtils.results.results import loadResults, splitOverParameter, whereParametersEqual, getBest
from experiment.tools import iterateDomains, parseCmdLineArgs
from analysis.matplotlib import displayOrSave, setFonts
from experiment import ExperimentModel

setFonts(plot_size='large')

# how much space should there be between algorithm "lanes"
FIDELITY = 200

ALG_ORDER = list(reversed([ 'ESARSA', 'SARSA', 'QLearning' ]))

def createEmptyList(shape, depth=0):
    out = None

    if len(shape) <= depth:
        return out

    out = [createEmptyList(shape, depth+1) for _ in range(shape[depth])]
    return out

def generatePlot(exp_paths):
    exp = ExperimentModel.load(exp_paths[0])
    results = loadResults(exp, 'step_return.csv')

    results_dict = splitOverParameter(results, 'tile-coder')

    # two new streams over results
    mine = results_dict['mine']
    tiles3 = results_dict['tiles3']

    # figure out what values of "tiles" we swept
    tiles = splitOverParameter(mine, 'tiles').keys()
    # figure out what values of "tilings" we swept
    tilings = splitOverParameter(mine, 'tilings').keys()

    f, axes = plt.subplots(len(tiles), len(tilings))

    # create an empty (2, tiles, tilings) shape array
    data = createEmptyList([2, len(tiles), len(tilings)])
    for tiles_idx, num_tiles in enumerate(tiles):
        for tilings_idx, num_tilings in enumerate(tilings):
            mine_result = whereParametersEqual(mine, { 'tiles': num_tiles, 'tilings': num_tilings })
            mine_result = getBest(mine_result, prefer='big')

            tiles3_result = whereParametersEqual(tiles3, { 'tiles': num_tiles, 'tilings': num_tilings })
            tiles3_result = getBest(tiles3_result, prefer='big')

            mine_data = [np.mean(curve) for curve in mine_result.load()]
            tiles3_data = [np.mean(curve) for curve in tiles3_result.load()]

            data[0][tiles_idx][tilings_idx] = mine_data
            data[1][tiles_idx][tilings_idx] = tiles3_data

    min_perf = np.min(data)
    max_perf = np.max(data)
    for rep, rep_name in enumerate(['Mine', 'Tiles3']):
        for tiles_idx, num_tiles in enumerate(tiles):
            for tilings_idx, num_tilings in enumerate(tilings):
                performance = data[rep][tiles_idx][tilings_idx]
                color = 'blue' if rep == 0 else 'red'

                kde = gaussian_kde(performance)
                lo = 0.95 * min_perf
                hi = 1.05 * max_perf
                dist_space = np.linspace(lo, hi, FIDELITY)
                dist = kde(dist_space)
                # dist = minMaxScale(kde(dist_space)) * DIST_HEIGHT
                axes[tiles_idx][tilings_idx].plot(dist_space, dist, label=rep_name, linewidth=2.0, color=color)
                axes[tiles_idx][tilings_idx].fill_between(dist_space, np.zeros(FIDELITY), dist, color=color, alpha=0.2)

                axes[tiles_idx][tilings_idx].set_xlim((0.95 * min_perf, 1.05 * max_perf))
                # axes[tiles_idx][tilings_idx].set_xlabel('Reward')

                title = f'({num_tiles} tiles, {num_tilings} tilings)'
                axes[tiles_idx][tilings_idx].set_title(title)

    plt.legend()

    return f, axes

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    for domain in iterateDomains(path):
        exp_paths = domain.exp_paths
        save_path = domain.save_path
        f, axes = generatePlot(exp_paths)

        file_name = os.path.basename(__file__).replace('.py', '').replace('_', '-')
        displayOrSave(f, should_save, save_path, domain.name, file_name, save_type, width=13, height=2 * (24/5))
