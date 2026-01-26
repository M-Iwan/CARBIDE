import os
import time
from itertools import product
from qcg.pilotjob.api.manager import LocalManager
from qcg.pilotjob.api.job import Jobs

user = 'miwan'

# Constants
python_path = f"/home/{user}/miniforge3/envs/pmd/bin/python3.11"
script_path = f"/home/{user}/Repos/PMD/src/CLI/qcg_job.sh"
input_dir = f"/home/{user}/Repos/PMD/data/carbide"
output_dir = f"/home/{user}/Repos/PMD/results/training"
sel_metric = 'HarmRS'
n_trials = 8
n_jobs = 1  # since we're running 96 in parallel...
max_concurrent_jobs = 5
sleep_interval = 30  # 5 minutes

os.makedirs(output_dir, exist_ok=True)

# LOOPS
dataset_types = ['primary', 'secondary']
pt_sets = ['Cvas', 'Card', 'Cred']
dpa_metrics = ['prr', 'ror', 'ic']

model_names = ['XGBClassifier', 'SVC', 'LogisticRegression', 'RandomForestClassifier']
desc_cols = ['RDKit', 'MACCS', 'Klek', 'Morgan', 'ChemBERTa', 'CDDD']
test_folds = [0, 1, 2, 3, 4]

# QCG-PilotJob

manager = LocalManager()
pending_jobs = []

jobs = Jobs()

for combination in product(dataset_types, pt_sets, dpa_metrics, model_names, desc_cols, test_folds):
    dataset_type, pt_set, dpa_metric, model_name, desc_col, test_fold = combination
    args = [input_dir, output_dir, dataset_type, pt_set, dpa_metric, model_name,
            desc_col, sel_metric, str(n_trials), str(n_jobs), str(test_fold)]

    name = '_'.join(map(str, combination))
    log_dir = f"{output_dir}/{dataset_type}_{pt_set}_{dpa_metric}/{model_name}/{desc_col}/logs"
    os.makedirs(log_dir, exist_ok=True)

    cmd = [script_path] + args

    job_dc = {
        'name': name,
        'exec': '/bin/bash',
        'args': cmd,
        'stdout': f"{log_dir}/tf_{test_fold}.out",
        'stderr': f"{log_dir}/tf_{test_fold}.err",
        'numCores': {"exact": n_jobs}
    }

    pending_jobs.append(job_dc)

running_jobs = []
while pending_jobs or running_jobs:
    # if slots available
    while pending_jobs and len(running_jobs) < max_concurrent_jobs:
        job_to_submit = pending_jobs.pop(0)
        jobs = Jobs()
        jobs.add(job_to_submit)
        manager.submit(jobs)
        running_jobs.append(job_to_submit['name'])
        print(f"Submitted job {job_to_submit['name']} ({len(running_jobs)}/{max_concurrent_jobs})")

    print(f"{len(running_jobs)} jobs running. Sleeping {sleep_interval}s...")
    time.sleep(sleep_interval)

    still_running = []
    for job_name in running_jobs:
        status = manager.status(job_name)['jobs'][job_name]['data']['status']
        if status not in ['SUCCEED', 'FAILED', 'CANCELED']:
            still_running.append(job_name)
    running_jobs = still_running

print('Submitting jobs')
manager.submit(jobs)
manager.wait4all()
manager.finish()
print('All jobs completed')