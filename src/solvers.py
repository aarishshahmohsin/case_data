import numpy as np
import random
from gurobipy import Model as GurobiModel, GRB, Env
import time
import gc
from constants import (
    epsilon_N,
    epsilon_P,
    epsilon_R,
    TIME_LIMIT,
    PRINT_OUTPUT,
    RAM_LIMIT,
)


def separating_hyperplane(P, N, eps_P, eps_N, eps_R, theta, lamb, num_trials=10000, seed=0):
    """
    Finds the initial separating hyperplane using the provided algorithm.

    Args:
        P (numpy.ndarray): Set of positive samples (numpy arrays).
        N (numpy.ndarray): Set of negative samples (numpy arrays).
        eps_P (float): Parameter for positive samples.
        eps_N (float): Parameter for negative samples.
        eps_R (float): Regularization parameter.
        theta (float): Scaling parameter.
        lamb (float): Lambda parameter for the optimization.
        num_trials (int): Number of random trials.

    Returns:
        tuple: Optimal hyperplane (w, c, reach), where w is the normal vector, c is the bias, and reach is the number of true positives.
    """

    np.random.seed(seed)

    dim = P.shape[1]  # Dimension of the feature space
    L = -np.inf
    best_h = None

    for _ in range(num_trials):
        # Choose a random unit vector w
        w = np.random.randn(dim)
        w /= np.linalg.norm(w)

        # Choose a random point c in the unit hypercube
        c = np.random.uniform(0, 1, dim)
        c = -np.dot(w, c)

        # Compute x_tilde and y_tilde arrays
        distances_P = np.dot(P, w) - c
        distances_N = np.dot(N, w) - c

        x_tilde = np.where(distances_P >= eps_P, 1, 0)
        y_tilde = np.where(distances_N > -eps_N, 1, 0)

        # Compute V_tilde
        V_tilde = max(
            0, ((theta - 1) * np.sum(x_tilde) + theta * np.sum(y_tilde) + theta * eps_R)
        )

        # Compute L_tilde
        L_tilde = np.sum(x_tilde) - V_tilde * lamb

        # Update L and best_h
        if L_tilde > L:
            L = L_tilde
            best_h = (w, c, np.sum(x_tilde))

    return best_h


def gurobi_solver(
    *,
    theta,
    theta0,
    theta1,
    P,
    N,
    epsilon_P=epsilon_P,
    epsilon_N=epsilon_N,
    epsilon_R=epsilon_R,
    lambda_param=None,
    dataset_name='random_name',
    run=True,
    seed=0,
    cuts=-1,
    method=-1,
):
    """
    Solves the wide-reach classification problem for given positive and negative samples.

    Parameters:
        theta (float): Precision threshold.
        P (numpy.ndarray): Feature matrix of positive samples (n_positive, n_features).
        N (numpy.ndarray): Feature matrix of negative samples (n_negative, n_features).

    Returns:
        dict: Contains the reach, hyperplane parameters, bias, and precision violation, or an error message.
    """
    X = np.vstack((P, N))

    # Update indices for P and N after combining
    num_positive = P.shape[0]

    P_indices = range(num_positive)
    N_indices = range(num_positive, X.shape[0])
    # Parameters
    if not lambda_param:
        lambda_param = (num_positive + 1) * theta1

    initial_h = separating_hyperplane(
        P, N, epsilon_P, epsilon_N, epsilon_R, theta, lambda_param, num_trials=10000, seed=seed
    )
    # print(initial_w, initial_c)

    # Create the Gurobi model
    model = GurobiModel("Wide-Reach Classification")
    env = Env()

    if TIME_LIMIT:
        model.setParam("TimeLimit", TIME_LIMIT)
    if not PRINT_OUTPUT:
        model.setParam("OutputFlag", 0)
    # if RAM_LIMIT:
    #     model.setParam("MemLimit", RAM_LIMIT)

    # Decision variables
    x = model.addVars(num_positive, vtype=GRB.BINARY, name="x")
    y = model.addVars(len(N_indices), vtype=GRB.BINARY, name="y")
    w = model.addVars(X.shape[1], lb=-GRB.INFINITY, name="w")
    c = model.addVar(lb=-GRB.INFINITY, name="c")
    V = model.addVar(lb=0, name="V")


    # Adjusting hyperparameters 
    model.setParam("Seed", seed)  # Fixed random seed
    # model.setParam("MIPFocus", 0) # Balanced search
    # model.setParam("MIPGap", 0.0001)  # Optimality gap
    # model.setParam("MIPGapAbs", 0.0001)  # Absolute gap
    # model.setParam("Presolve", 1)  # Moderate presolve
    model.setParam('Cuts', cuts)
    model.setParam("Method", method)  # Interior Point (Barrier)
    # above 0, 1, 2, 3


    if initial_h is not None:
        init_w, init_c, reach = initial_h
        distances_P = np.dot(P, init_w) - init_c
        distances_N = np.dot(N, init_w) - init_c

        xs = distances_P >= epsilon_P
        ys = distances_N > -epsilon_N
        assert xs.sum() == reach
        for i in range(num_positive):
            x[i].Start = int(xs[i])
        for i in range(len(N_indices)):
            y[i].Start = int(ys[i])

    # Objective: Maximize the reach minus penalty for precision violation
    model.setObjective(sum(x[i] for i in P_indices) - lambda_param * V, GRB.MAXIMIZE)

    # Constraint: Precision constraint violation
    model.addConstr(
        V
        >= (theta - 1) * sum(x[i] for i in P_indices)
        + theta * sum(y[j] for j in range(len(N_indices)))
        + theta * epsilon_R,
        "PrecisionConstraint",
    )

    # Constraints: Classification constraints for positive samples
    for i, p_idx in enumerate(P_indices):
        model.addConstr(
            x[i]
            <= 1 + sum(w[d] * X[p_idx, d] for d in range(X.shape[1])) - c - epsilon_P,
            name=f"Positive_{i}",
        )

    # Constraints: Classification constraints for negative samples
    for j, n_idx in enumerate(N_indices):
        model.addConstr(
            y[j] >= sum(w[d] * X[n_idx, d] for d in range(X.shape[1])) - c + epsilon_N,
            name=f"Negative_{j}",
        )

    model.write(f"{dataset_name}.lp")
    # Solve the model
    if run:
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        # Check and return results
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            results = {
                "Reach": sum(x[i].x for i in P_indices),
                "Hyperplane w": [w[d].x for d in range(X.shape[1])],
                "Bias c": c.x,
                "X": [x[d].x for d in range(len(P))],
                "Y": [y[d].x for d in range(len(N))],
                "Precision Violation V": V.x,
                "Node Count": model.NodeCount,
                "Time taken": end_time - start_time,
            }
            # return results
        else:
            results = {"Error": "No optimal solution found."}

        # dispose everything
        model.reset()
        model.dispose()
        env.dispose()

        del model
        del env

        gc.collect()

        return results

    else:
        return None


