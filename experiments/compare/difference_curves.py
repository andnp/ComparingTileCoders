import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd() + '/src')

from analysis.learning_curve import lineplot
from PyExpUtils.results.results import loadResults, whereParametersEqual, splitOverParameter, getBest
from experiment.tools import iterateDomains, parseCmdLineArgs
from analysis.matplotlib import displayOrSave, setFonts
from experiment import ExperimentModel

setFonts(plot_size='large')

def generatePlot(ax, exp_paths):
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

    for num_tiles in tiles:
        for num_tilings in tilings:
            mine_result = whereParametersEqual(mine, { 'tiles': num_tiles, 'tilings': num_tilings })
            mine_result = getBest(mine_result, prefer='big')

            tiles3_result = whereParametersEqual(tiles3, { 'tiles': num_tiles, 'tilings': num_tilings })
            tiles3_result = getBest(tiles3_result, prefer='big')

            d_curves = []
            mine_curves = mine_result.load()
            tiles3_curves = tiles3_result.load()

            min_len = min(len(mine_curves), len(tiles3_curves))
            print(min_len)
            for i in range(min_len):
                d = mine_curves[i] - tiles3_curves[i]
                d_curves.append(d)

            mean = np.mean(d_curves, axis=0)
            stderr = np.std(d_curves, axis=0, ddof=1) / np.sqrt(len(d_curves))

            lineplot(ax, mean, stderr=stderr, label=f'({num_tiles}, {num_tilings})')

    ax.set_xlim(0, 500)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax.set_ylabel(f'd = Mine - Tiles3 \n Bigger is better')
    plt.legend()

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    for domain in iterateDomains(path):
        exp_paths = domain.exp_paths
        save_path = domain.save_path

        f, axes = plt.subplots(1)
        f.suptitle(f'{domain.name}')

        generatePlot(axes, exp_paths)

        file_name = os.path.basename(__file__).replace('.py', '').replace('_', '-')
        displayOrSave(f, should_save, save_path, domain.name, file_name, save_type, width=5, height=(24/5))
