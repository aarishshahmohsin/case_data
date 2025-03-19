from real_datasets import BreastCancerDataset, WineQualityRedDataset, WineQualityWhiteDataset, SouthGermanCreditDataset, CropMappingDataset
import os
import numpy as np
from synthetic_datasets import ClusterDataset, TwoClusterDataset, DiffusedBenchmark, PrismDataset, TruncatedNormalPrism
from solvers import gurobi_solver, separating_hyperplane
from constants import epsilon_P, epsilon_N, epsilon_R
datasets = {
    "Breast Cancer": BreastCancerDataset(),
    "Crop Mapping": CropMappingDataset(),
    "Wine Quality Red": WineQualityRedDataset(),
    "Wine Quality White": WineQualityWhiteDataset(),
    "South German Credit": SouthGermanCreditDataset(),
    "Cluster 8": ClusterDataset(d=8),
    "Two Cluster 8": TwoClusterDataset(d=8),
    "Cluster": ClusterDataset(d=11),
    "Two Cluster": TwoClusterDataset(d=11),
    "Diffused Benchmark": DiffusedBenchmark(),
    "Prism": PrismDataset(),
    "Truncated Normal Prism": TruncatedNormalPrism(),
}

times = 5
SEEDS = [42, 123, 0, 2024, 314159, 271828]
# SEEDS = [42, 123, 0, 2024, 314159, 271828, 161803, 8675309, 13579, 24680]

for cut in range(-1, 4):
    for method in range(-1, 4):
        print("new cut: ", cut)
        if method == -1 and (cut == -1 or cut == 0):
            os.system('python /tmp/f/case_data/sender.py')
        final_res = []
        for dataset_name, dataset in datasets.items():
            P, N = dataset.generate()
            t0, t1, t, l = dataset.params()
            # print(t1)
            for i in range(times):
                # print(i)
                # print(l)
                # w, c, r = separating_hyperplane(P, N, epsilon_P, epsilon_N, epsilon_R, t, l, num_trials=10000)
                # # print(c) 
                # reach = np.dot(P, w) - c >= epsilon_P
                # print(np.sum(reach))
                # # xs ≤ 1 + sT w − c − P
                # res_cplex = cplex_solver(theta=t, theta0=t0, theta1=t1, P=P, N=N, lambda_param=l)
                # print("cplex", dataset_name, res_cplex['Reach'])
                res_gurobi = gurobi_solver(theta=t, theta0=t0, theta1=t1, P=P, N=N, lambda_param=l, seed=SEEDS[i], cuts=cut, method=method)
                final_res.append(cut, method, SEEDS[i], "gurobi", dataset_name, res_gurobi['Reach'], res_gurobi['Time taken'])
                print(SEEDS[i], "gurobi", dataset_name, res_gurobi['Reach'], res_gurobi['Time taken'])
                # print(res_gurobi)
                # final_res.append([res_gurobi, res_cplex])
                # print([res_gurobi['Reach'], res_cplex['Reach']])
                # print(r)
            # results[dataset_name] = final_res
        print('new iteration', cut, method)
        print(final_res)
            