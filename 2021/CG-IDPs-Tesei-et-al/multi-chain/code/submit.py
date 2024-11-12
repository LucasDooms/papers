from analyse import initProteins
from collections import defaultdict
from typing import Dict
from jinja2 import Template
import pandas as pd
import signac
import subprocess


# init project
project = signac.init_project()


# DEFINE PARAMETERS HERE

protein_names = ['FUS']
temperatures = [323, 360, 380, 400]
model = 'M1'
seed = 1431455135312
method = "resize"
Nsteps = int(2e7)
walltime = {'d': 1, 'h': 0, 'm': 0, 's': 0}

# END OF PARAMETERS


def to_seconds(wt: Dict[str, int]) -> int:
    convert = defaultdict(default_factory=0, **{'s': 1, 'm': 60, 'h': 60 * 60, 'd': 24 * 60 * 60})
    return sum([convert[key] * val for key, val in wt.items()])

def slurm_format(seconds: int) -> str:
    s = seconds % 60
    seconds //= 60
    m = seconds % 60
    seconds //= 60
    h = seconds % 24
    seconds //= 24
    d = seconds
    if d > 99:
        raise Exception(f"Walltime ({d} days, {h} hours, {m} minutes, {s} seconds) too large, cannot format to DD-HH-MM-SS")
    return f"{d:02d}-{h:02d}:{m:02d}:{s:02d}"

walltime_seconds = to_seconds(walltime)
walltime_formatted = slurm_format(walltime_seconds)

proteins_db = initProteins()
for name, prot in proteins_db.loc[protein_names].iterrows():
    for temp in temperatures:
        statepoint = dict(name=name, temp=temp, model=model, seed=seed, Nsteps=Nsteps, method=method, walltime_dict=walltime, walltime=walltime_seconds)
        job = project.open_job(statepoint)
        job.init()


with open("submit_template.sh", "r") as file:
    contents = file.read()
    submission = Template(contents)

# create and submit cluster jobs
for job in project:
    name = job.sp.name
    temp = job.sp.temp
    id = job.id
    script_name = f"{name:s}_{temp:d}_{id}.sh"

    with open(script_name, 'w') as submit:
        submit.write(
            submission.render(
                name=name,
                temp=f'{temp:d}',
                walltime_formatted=walltime_formatted,
                id=id
            )
        )

    subprocess.run(['sbatch', script_name])
