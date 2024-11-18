

import itertools
import numpy as np

def min_bins(weights, C):# n is the number of bins
    n  = len(weights)
    w_s = sum(weights)
    if w_s > n * C:
        print('increase capacity or the number of bins')
        return
    while w_s < n * C :
        n -= 1
        if w_s > n * C :
            return n + 1
        elif w_s == n * C:
            return n
        else:
            continue



def create_y(num_items):
    """
    Create an array of arrays Y with increasing numbers of 1s in each row.

    Parameters:
    num_items (int): The number of items (length of each array).

    Returns:
    list: A list of arrays with increasing numbers of 1s.
    """
    Y = []
    for i in range(1, num_items + 1):
        # Create a binary array with the first i elements as 1, the rest as 0
        arr = np.zeros(num_items, dtype=int)
        arr[:i] = 1
        Y.append(arr.tolist())

    return Y

def Binary_Constraint(n, m):
    """
    Generates all possible configurations of x_{ij} and filters out only those
    combinations where each item i is assigned to exactly one bin j.

    Parameters:
    n (int): Number of items.
    m (int): Number of bins.

    Returns:
    list: A list of valid configurations where each item is assigned to exactly one bin.
          Each configuration is an (n, m) array of 0s and 1s.
    """

    # Generate all possible combinations of x_{ij} for each item in each bin
    all_combinations = list(itertools.product([0, 1], repeat=n * m))
    valid_combinations = []

    # Iterate over all possible combinations
    for combination in all_combinations:
        # Reshape the combination into an (n, m) matrix
        x_matrix = np.array(combination).reshape(n, m)

        # Check if each row (each item) has exactly one '1'
        if np.all(np.sum(x_matrix, axis=1) == 1):
            valid_combinations.append(x_matrix)

    return valid_combinations


def Capacity_Constraint(y, weights, C):
    """
    Filters configurations based on bin capacity constraints using a single set of y_j values.

    Parameters:
    y (list): A single list of y_j values for the current configuration (length m).
    weights (list): List of weights for each item (length n).
    C (int): Capacity of each bin.

    Returns:
    list: A list of configurations that satisfy the bin capacity constraint.
    """

    n = len(weights)
    m = min_bins(weights, C) # Assumption: No. of items = no. of bins
    valid_configs = Binary_Constraint(n,m)

    valid_capacity_configs = []

    # Iterate over each configuration
    for config in valid_configs:
        satisfies_capacity = True  # Flag to check if the current config satisfies capacity

        # Check each bin for the capacity constraint using the provided y values
        for j in range(config.shape[1]):  # Assuming config is an (n, m) matrix
            # Compute the total weight in bin j for the current configuration
            total_weight_in_bin = np.sum(weights[i] * config[i, j] for i in range(config.shape[0]))

            # Check if the total weight in bin j exceeds the bin capacity using provided y_j
            if total_weight_in_bin > C * y[j]:
                satisfies_capacity = False
                break  # No need to check further bins if one fails

        # Add configuration if it satisfies the capacity constraint
        if satisfies_capacity:
            valid_capacity_configs.append(config)

    return valid_capacity_configs


def brute_force( weights, C):
    """
    Brute force search to find the first valid configuration of x_{ij}
    that satisfies the Binary and Capacity Constraints.

    Parameters:
    weights (list): List of weights for each item (length n).
    C (int): Capacity of each bin.

    Returns:
    tuple: The first valid configuration and the number of bins used (1s in y_j).
    """ 
    n = len(weights) # number of items
    # Step 1: Initialize Y using create_y
    Y = create_y(n)

    # Step 2: Iterate over each y_j in Y
    for y_j in Y:

        # Step 3: Find the possible combination of x_{ij} passing through Binary_Constraint and Capacity_Constraint
        valid_capacity_configs = Capacity_Constraint(y_j, weights, C)

        # Step 4: If there is a valid configuration, return it and the number of bins used (1s in y_j)
        if valid_capacity_configs:
            # Return the first valid configuration and number of bins used
            return valid_capacity_configs[0], sum(y_j)

    # If no valid configuration is found
    print("No Combination Possible")
    return None, 0

weights = [3, 2, 3, 2, 1]  # Weights of 4 items
C =  max(weights) # Bin capacity
brute_force(weights, C)
