import os
from typing import Any, Dict
import matplotlib.pyplot as plt

def setFonts(plot_size='large'):
    if plot_size == 'small':
        SMALL_SIZE = 10
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 13

    elif plot_size == 'large':
        SMALL_SIZE = 14
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 18

    else:
        raise NotImplementedError()

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def paramsToStr(params: Dict[str, Any]):
    parts = [f'{params[key]}' for key in params]

    return '_'.join(parts)

def displayOrSave(
        f,
        should_save: bool,
        save_path: str,
        domain_name: str,
        plot_name: str,
        save_type: str,
        params: Dict[str, Any] = {},
        width: float = 8,
        height: float = 24/5,
    ):

    if not should_save:
        plt.show()
        exit()

    save_path = f'{save_path}/{plot_name}'
    os.makedirs(save_path, exist_ok=True)

    name = domain_name
    if len(params):
        name += f'_{paramsToStr(params)}'

    f.set_size_inches((width, height), forward=True)
    plt.savefig(f'{save_path}/{name}.{save_type}', dpi=300, bbox_inches='tight')
