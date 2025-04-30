# partial_sampling_spatially_structured_population
Repository for my lab immersion on spatially structured populations with partial updates.

Using the formalism from  ["Bridging Wright–Fisher and Moran models"](https://www.sciencedirect.com/science/article/pii/S0022519324003151?via%3Dihub).

# Code overview

**slurm_jobs** contains scripts for SCITAS clusters.

**utils** contains the more elementary scripts:

- `graph_generation.py`: creates the migration matrices based on the specified type and parameters

- `graph_simulation.py`: launches graph-based simulations

- `misc.py`: miscellaneous, among which a failed attempt to compute the theoretical mean extinction time (from equation S32 of the paper)

- `wm_mat.py`: computing the fixation probability using the transition (well-mixed case)

- `wm_sim.py`: launches simulations in the well-mixed case

`run_single.py` and `slurm_main.py` are used to generate and store simulation data (see part below on how to use them).

`visualization.py` is used to produce plots.

`run.ipynb` is a notebook using all the functions to launch simulations and produce plots (the last part is not functional).


# Run simulations / computations

## slurm_main.py

Runs multiple simulations / computations of the specified type, sweeping the fitness in `np.logspace(log_s_min, log_s_max, num)` and writes a .JSON file in the specified results directory.

You can run it as a script (which you must modify to specify `prefix`, `results_dir`, `num`; useful when you want to run them on SCITAS clusters) or as a function (see examples in `run.ipynb`).

>Make sure the results directory is created before running

>It is not possible to store the fixation/extinction times. (use `run_single.py` instead)

### Function

**Arguments:**

- `prefix`: str, beginning of the filename, e.g. 'simA'

- `results_dir`: str, path where the results should be written, e.g. 'results/simA/' (must end with a '/')

- `num`: int, number of fitness values in the logspace array

- `type`: str, ['wm_sim', 'wm_mat', 'clique', 'cycle', 'line', 'star']

- `job_array_nb`: int, identifier (useful for running array jobs with slurm too), will be added in the filename

- `N`: int, number of individuals per deme

- `M`: int, number of updated individuals per deme

- `log_s_min`: int,

- `log_s_max`: int,

- (optional) `nb_trajectories`: int = 100, number of simulation runs

- (optional) `migration_rate`: float = 0.001,

- (optional) `nb_demes`: int = 1, number of demes,

- (optional) `alpha`: float = 1., migration assymetry (if type is 'cycle', 'line' or 'star')

- (optional) `initial_node`: int = 0, node where the first mutant spawns (between 0 and nb_demes - 1), or makes an average if initial_node = 'avg' (for line and star graphs)


**Returns nothing, writes `{prefix}_{type}_{job_array_nb}_{parameters}.json` in the results directory**



### Script

Template:

`py slurm_main.py [type] [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

The arguments in parentheses depend on the chosen type:

- For clique graphs: `py slurm_main.py clique [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes)`

- For cycle graphs: `py slurm_main.py cycle [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes) (alpha)`

- For star graphs: `py slurm_main.py star [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

- For line graphs: `py slurm_main.py line [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

> You can choose to have the averaged fixation probability over the possible initial nodes if `initial node = 'avg'` (for line and star graphs). 

- For well-mixed simulations: `py slurm_main.py wm_sim [job_array_nb] [N] [M] [log_s_min] [log_s_max] [nb_trajectories]`

- For well-mixed matrix computations: `py slurm_main.py wm_mat [job_array_nb] [N] [M] [log_s_min] [log_s_max]`

## run_single.py

Runs multiple simulations / computations of the specified type, using a single fitness value and writes a .JSON file in the specified results directory

You can run it as a script (which you must modify to specify `prefix` or `results_dir`; useful when you want to run them on SCITAS clusters) or as a function (see examples in `run.ipynb`)

>Make sure the results directory is created before running

>Here, it is possible to store the fixation/extinction times.


### Function

Runs multiple simulations / computations of the specified type, sweeping the fitness in np.logspace(log_s_min, log_s_max, num) and writes a .JSON file in the specified results directory

Make sure the results directory is created before running

**Arguments:**

- `prefix`: str, beginning of the filename, e.g. 'simA'

- `results_dir`: str, path where the results should be written, e.g. 'results/simA/' (must end with a '/')

- `type`: str, ['wm_sim', 'wm_mat', 'clique', 'cycle', 'line', 'star']

- `job_array_nb`: int, identifier (useful for running array jobs with slurm too), will be added in the filename

- `N`: int, number of individuals per deme

- `M`: int, number of updated individuals per deme

- `s`: float, relative fitness

- (optional) `nb_trajectories`: int = 100, number of simulation runs

- (optional) `migration_rate`: float = 0.

- (optional) `nb_demes`: int = 1, number of demes,

- (optional) `alpha`: float = 1., migration assymetry (if type is 'cycle', 'line' or 'star')

- (optional) `initial_node`: int = 0, node where the first mutant spawns (between 0 and nb_demes - 1), or makes an average if initial_node = 'avg' (for line and star graphs)


**Returns nothing, writes `{prefix}_single_{type}_{job_array_nb}_{parameters}.json` in the results directory**


### Script

Template:

`py run_single.py [type] [job_array_nb] [N] [M] [s] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

The arguments in parentheses depend on the chosen type:

- For clique graphs: `py run_single.py clique [job_array_nb] [N] [M] [s] [nb_trajectories] (migration_rate) (nb_demes)`

- For cycle graphs: `py run_single.py cycle [job_array_nb] [N] [M] [s] [nb_trajectories] (migration_rate) (nb_demes) (alpha)`

- For star graphs: `py run_single.py star [job_array_nb] [N] [M] [s] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

- For line graphs: `py run_single.py line [job_array_nb] [N] [M] [s] [nb_trajectories] (migration_rate) (nb_demes) (alpha) (initial node)`

- For well-mixed simulations: `py run_single.py wm_sim [job_array_nb] [N] [M] [s] [nb_trajectories]`

- For well-mixed matrix computations: `py run_single.py wm_mat [job_array_nb] [N] [M] [s]`


