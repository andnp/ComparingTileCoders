# Comparing Tile-coder Implementations

## Setting up repo
**This codebase only works with python 3.6 and above.**

Packages are stored in a `requirements.txt` file.
To install:
```
pip install -r requirements.txt
```

On machines that you do not have root access to (like compute canada machines), you will need to install in the user directory.
You can do this with:
```
pip install --user -r requirements.txt
```
Or you need to set up a virtual environment:
```
virtualenv -p python3 env
```


---
## Step-by-step example

```bash
cd ComparingTileCoders
git pull # make sure you are up to date

# run the experiment for 100 runs
python scripts/local.py src/main.py ./ 100 experiments/compare/**/*.json

# plot your results (add "save" as an argument to this function to save the plot as a png)
python experiments/learning_curves.py
```


---
## Dependencies
This template repo depends on a few other shared libraries to make code-splitting and sharing a little easier (for me).
The documentation and source code can be found at the following links.
* [RLGlue](https://github.com/andnp/rlglue) - my own minimal implementation of RLGlue
* [PyExpUtils](https://github.com/andnp/pyexputils) - a library containing the experiment running framework
* [PyFixedReps](https://github.com/andnp/pyfixedreps) - a few fixed representation algorithms implemented in python (e.g. tile-coding, rbfs, etc.)


---
## Organization Patterns

### Experiments
All experiments are described as completely as possible within static data files.
I choose to use `.json` files for human readability and because I am most comfortable with them.
These are stored in the `experiments` folder, usually in a subdirectory with a short name for the experiment being run (e.g. `experiments/idealH` would specify an experiment that tests the effects of using h*).

Experiment `.json` files look something like:
```jsonc
{
    "agent": "gtd2", // <-- name of your agent. these names are defined in agents/registry.py
    "problem": "randomwalk", // <-- name of the problem you're solving. these are defined in problems/registry.py
    "metaParameters": { // <-- a dictionary containing all of the meta-parameters for this particular algorithm
        "alpha": [1, 0.5, 0.25], // <-- sweep over these 3 values of alpha
        "beta": 1.0, // <-- don't sweep over beta, always use 1.0
        "use_ideal_h": true,
        "lambda": [0.0, 0.1]
    }
}
```

### Problems
I define a **problem** as a combination of:
1) environment
2) representation
3) target/behavior policies
4) number of steps
5) gamma
6) starting conditions for the agent (like in Baird's)

### results
The results are saved in a path that is defined by the experiment definition used.
The configuration for the results is specified in `config.json`.
Using the current `config.json` yields results paths that look like:
```
<base_path>/results/<experiment short name>/<agent name>/<parameter values>/errors_summary.npy
```
Where `<base_path>` is defined when you run an experiment.

### src
This is where the source code is stored.
The only `.py` files it contains are "top-level" scripts that actually run an experiment.
No utility files or shared logic at the top-level.

**agents:** contains each of the agents.
Preferably, these would be one agent per file.

**analysis:** contains shared utility code for analysing the results.
This *does not* contain scripts for analysing results, only shared logic (e.g. plotting code or results filtering).

**environments:** contains minimal implementations of just the environment dynamics.

**utils:** various utility code snippets for doing things like manipulating file paths or getting the last element of an array.
These are just reusable code chunks that have no other clear home.
I try to sort them into files that roughly name how/when they will be used (e.g. things that manipulate files paths goes in `paths.py`, things that manipulate arrays goes in `arrays.py`, etc.).

### clusters
This folder contains the job submission information that is needed to run on a cluster.
These are also `.json` files that look like:
```jsonc
{
    "account": "which compute canada account to use",
    "time": "how much time the job is expected to take",
    "nodes": "the number of cpu cores to use",
    "memPerCpu": "how much memory one parameter setting requires", // doesn't need to change
    "tasksPerNode": "how many parameter settings to run in serial on each cpu core"
}
```

Some quick terminology (that I made up and is kinda bad):
* **node**: a CPU core
* **task**: a single call to the experiment entry file (e.g. `src/main.py`). Generally only runs one parameter setting for a single run.
* **job**: a compute canada job (contains many tasks and run across multiple nodes).

The `nodes` setting determines the number of CPU cores for the job to request.
These CPU cores may not all be on the same server node and most likely will be split across several server nodes.
The job scheduling script bundled with this template repo will handle distributing jobs across multiple server nodes in the way recommended by compute canada support.

The `tasksPerNode` sets up the number of processes (calls to the experiment entry file) to be lined up per node requested.
If you request `nodes=16`, then 16 jobs will be run in **parallel**.
If you request `tasksPerNode=4`, then each node will run 4 tasks in **serial**.
In total, 64 tasks will be scheduled for one compute canada job with this configuration.
If there are 256 total tasks that need to be run, then 4 compute canada jobs will be scheduled.
