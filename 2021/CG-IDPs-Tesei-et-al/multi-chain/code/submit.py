from analyse import initProteins
from collections import defaultdict
from typing import Dict
from pathlib import Path
from jinja2 import Template
import pandas as pd
import subprocess

# DEFINE PARAMETERS HERE

protein_names = ['ht40']
temperatures = [323, 360, 380, 400]
model = 'M1'

# set to all zeros to disable walltime (and remove it from submit_template.sh)
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

with open("submit_template.sh", "r") as file:
    contents = file.read()
    submission = Template(contents)

residues = pd.read_csv('residues.csv', float_precision='round_trip').set_index('three')
residues.lambdas = residues[model]
residues.to_csv('residues.csv')

for name, prot in proteins_db.loc[protein_names].iterrows():
    base_path = Path(name)
    base_path.mkdir(exist_ok=True)

    for temp in temperatures:
        temp_path = base_path / f'{temp:d}'
        temp_path.mkdir(exist_ok=True)

        script_name = f"{name:s}_{temp:d}.sh"

        with open(script_name, 'w') as submit:
            submit.write(
                submission.render(
                    name=name,
                    temp=f'{temp:d}',
                    walltime_formatted=walltime_formatted,
                    walltime_seconds=walltime_seconds
                )
            )

        subprocess.run(['sbatch', script_name])
