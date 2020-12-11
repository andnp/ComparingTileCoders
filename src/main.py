import numpy as np
import time
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from PyExpUtils.utils.Collector import Collector
from src.utils.rlglue import OneStepWrapper

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    exit(1)

exp = ExperimentModel.load(sys.argv[1])
idx = int(sys.argv[2])

max_steps = exp.max_steps
run = exp.getRun(idx)

collector = Collector()
# set random seeds accordingly
np.random.seed(run)

Problem = getProblem(exp.problem)
problem = Problem(exp, idx)

agent = problem.getAgent()
env = problem.getEnvironment()

wrapper = OneStepWrapper(agent, problem.getGamma(), problem.rep)

glue = RlGlue(wrapper, env)

# Run the experiment
glue.start()
start_time = time.time()
episode = 0

for step in range(exp.max_steps):
    _, _, _, t = glue.step()

    if t:
        episode += 1
        glue.start()

        # collect an array of rewards that is the length of the number of steps in episode
        # effectively we count the whole episode as having received the same final reward
        collector.concat('step_return', [glue.total_reward] * glue.num_steps)

        # compute the average time-per-step in ms
        avg_time = 1000 * (time.time() - start_time) / step
        print(episode, step, glue.total_reward, f'{avg_time:.4}ms')

        glue.total_reward = 0
        glue.num_steps = 0

collector.fillRest('step_return', exp.max_steps)
collector.collect('time', time.time() - start_time)
collector.collect('feature_utilization', np.count_nonzero(agent.w) / np.product(agent.w.shape))

# import matplotlib.pyplot as plt
# from src.utils.plotting import plot
# fig, ax1 = plt.subplots(1)

# collector.reset()
# return_data = collector.getStats('step_return')
# plot(ax1, return_data)
# ax1.set_title('Return')

# plt.show()
# exit()

from PyExpUtils.results.backends.csv import saveResults
from PyExpUtils.utils.arrays import downsample

for key in collector.run_data:
    data = collector.run_data[key]
    # heavily downsample the data to reduce storage costs
    # we don't need all of the data-points for plotting anyways
    # method='window' returns a window average
    # method='subsample' returns evenly spaced samples from array
    # num=1000 makes sure final array is of length 1000
    # percent=0.1 makes sure final array is 10% of the original length (only one of `num` or `percent` can be specified)
    data = downsample(data, num=500, method='window')

    saveResults(exp, idx, key, data, precision=2)
