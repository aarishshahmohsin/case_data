import numpy as np
from real_datasets import (
    BreastCancerDataset,
    WineQualityRedDataset,
    WineQualityWhiteDataset,
    SouthGermanCreditDataset,
    CropMappingDataset,
)
from synthetic_datasets import (
    ClusterDataset,
    TwoClusterDataset,
    DiffusedBenchmark,
    PrismDataset,
    TruncatedNormalPrism,
)
from solvers import gurobi_solver
import os
import subprocess
import re

# Define the datasets
datasets = {
    "Breast Cancer": BreastCancerDataset(),
    "Wine Quality Red": WineQualityRedDataset(),
    "Wine Quality White": WineQualityWhiteDataset(),
    "South German Credit": SouthGermanCreditDataset(),
    "Crop Mapping": CropMappingDataset(),
    "Cluster 8": ClusterDataset(d=8),
    "Two Cluster 8": TwoClusterDataset(d=8),
    "Cluster": ClusterDataset(d=11),
    "Two Cluster": TwoClusterDataset(d=11),
    "Diffused Benchmark": DiffusedBenchmark(),
    "Prism": PrismDataset(),
    "Truncated Normal Prism": TruncatedNormalPrism(),
}

# Generate .lp files for all datasets
ds_list = []
for dataset_name, dataset in datasets.items():
    P, N = dataset.generate()
    theta_0, theta_1, theta, lambda_param = dataset.params()
    ds_name = "".join(dataset_name.lower().split(' '))
    ds_list.append(ds_name)
    gurobi_solver(
        theta=theta,
        theta0=theta_0,
        theta1=theta_1,
        P=P,
        N=N,
        lambda_param=lambda_param,
        dataset_name=ds_name,
        run=False
    )

# Define time limits
time_limit = 120
threads = 48
# tune_time_limit = 120

# Run the tuning process for all datasets
# command = f"grbtune LogToConsole=1 TuneTimeLimit={tune_time_limit} TimeLimit={time_limit} {' '.join([ds + '.lp' for ds in ds_list])}"
command = f"grbtune Threads={threads} LogToConsole=1  TimeLimit={time_limit} {' '.join([ds + '.lp' for ds in ds_list])}"

# Run the command and capture output
print(f"Running command: {command}")
process = subprocess.run(command, shell=True, text=True)
output = process.stdout



