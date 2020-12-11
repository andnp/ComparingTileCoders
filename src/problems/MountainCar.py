from problems.BaseProblem import BaseProblem
from environments.MountainCar import MountainCar as MCEnv
from PyFixedReps.TileCoder import TileCoder
from utils.tiles3 import Tiles3

class MountainCar(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = MCEnv(self.seed)
        self.actions = 3

        tc = self.params['tile-coder']

        if tc == 'mine':
            self.rep = TileCoder({
                'dims': 2,
                'tiles': self.params['tiles'],
                'tilings': self.params['tilings'],
                'input_ranges': [(-1.2, 0.5), (-0.07, 0.07)],
                'scale_output': True,
            })

        elif tc == 'tiles3':
            self.rep = Tiles3({
                'dims': 2,
                'tiles': self.params['tiles'],
                'tilings': self.params['tilings'],
                'input_ranges': [(-1.2, 0.5), (-0.07, 0.07)],
                'scale_output': True,
            })

        else:
            raise NotImplementedError()

        self.features = self.rep.features()
        self.gamma = 1.0
