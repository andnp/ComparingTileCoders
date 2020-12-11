import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.getcwd() + '/src')

from analysis.learning_curve import plotBest
from PyExpUtils.results.results import loadResults, whereParametersEqual, splitOverParameter, getBest
from experiment.tools import iterateDomains, parseCmdLineArgs
from analysis.matplotlib import displayOrSave, setFonts
from experiment import ExperimentModel

setFonts(plot_size='large')

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

    for tiles_idx, num_tiles in enumerate(tiles):
        for tilings_idx, num_tilings in enumerate(tilings):
            mine_result = whereParametersEqual(mine, { 'tiles': num_tiles, 'tilings': num_tilings })
            mine_result = getBest(mine_result, prefer='big')

            tiles3_result = whereParametersEqual(tiles3, { 'tiles': num_tiles, 'tilings': num_tilings })
            tiles3_result = getBest(tiles3_result, prefer='big')

            plotBest(mine_result, axes[tiles_idx][tilings_idx], color='blue', label='Mine')
            plotBest(tiles3_result, axes[tiles_idx][tilings_idx], color='red', label='Tiles3')

            axes[tiles_idx][tilings_idx].set_title(f'Tiles: {num_tiles} Tilings: {num_tilings}')

    axes[0][0].legend()

    return f, axes

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    for domain in iterateDomains(path):
        exp_paths = domain.exp_paths
        save_path = domain.save_path

        f, axes = generatePlot(exp_paths)
        f.suptitle(f'{domain.name}')


        file_name = os.path.basename(__file__).replace('.py', '').replace('_', '-')
        displayOrSave(f, should_save, save_path, domain.name, file_name, save_type, width=11, height=2 * (24/5))
