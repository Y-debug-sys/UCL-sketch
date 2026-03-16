import math
import numpy as np

from collections import Counter


def average_absolute_error(truth, predict):
    """
    Calculate the Average Absolute Error (AAE) between true and predicted values.
    This metric measures the mean absolute difference between actual and estimated frequencies.
    
    Args:
        truth: List or array of true frequency values
        predict: List or array of predicted/estimated frequency values
        
    Returns:
        Mean absolute error between true and predicted values
    """
    gt, et = np.array(truth), np.array(predict)
    return np.abs(et - gt).mean()


def average_relative_error(truth, predict):
    """
    Calculate the Average Relative Error (ARE) between true and predicted values.
    This metric measures the mean relative difference, normalized by the true value.
    
    Args:
        truth: List or array of true frequency values
        predict: List or array of predicted/estimated frequency values
        
    Returns:
        Mean relative error between true and predicted values
    """
    gt, et = np.array(truth), np.array(predict)
    return (np.abs(et - gt) / gt).mean()


def average_weighted_error(truth, predict):
    """
    Calculate the Average Weighted Error between true and predicted values.
    This metric weights the absolute differences by the true values.
    
    Args:
        truth: List or array of true frequency values
        predict: List or array of predicted/estimated frequency values
        
    Returns:
        Weighted mean absolute error between true and predicted values
    """
    gt, et = np.array(truth), np.array(predict)
    return (np.abs(et - gt) * gt).mean()


def weighted_mean_relative_difference(truth, predict):
    """
    Calculate the Weighted Mean Relative Difference (WMRD) between true and predicted distributions.
    This metric compares the frequency distributions of true and predicted values.
    
    Args:
        truth: List or array of true frequency values
        predict: List or array of predicted/estimated frequency values
        
    Returns:
        Weighted mean relative difference between the two distributions
    """
    wmrd1 = wmrd2 = 0
    # Count occurrences of each frequency value in both arrays
    gt_count = dict(Counter(truth))
    et_count = dict(Counter(predict))
    union_count = set(gt_count.keys()).union(set(et_count.keys()))

    for n in union_count:
        # Get count of frequency n in ground truth, default to 0 if not present
        try:
            n1 = gt_count[n]
        except:
            n1 = 0 

        # Get count of frequency n in estimate, default to 0 if not present
        try:
            n2 = et_count[n]
        except:
            n2 = 0 
        
        wmrd1 += abs(n1 - n2)  # Sum of absolute differences
        wmrd2 += ((n1 + n2) / 2)  # Sum of averages

    return wmrd1 / wmrd2  # Normalize the difference by the average counts


def entropy_absolute_error(truth, predict):
    """
    Calculate the Entropy Absolute Error between true and predicted frequency distributions.
    This metric computes the difference in entropy between the two distributions,
    measuring how much the uncertainty differs between true and predicted values.
    
    Args:
        truth: List or array of true frequency values
        predict: List or array of predicted/estimated frequency values
        
    Returns:
        Absolute difference in entropy between the two distributions
    """
    n_key = len(truth)  # Total number of items
    gt_epy = et_epy = 0  # Initialize entropies
    # Count occurrences of each frequency value in both arrays
    gt_count = dict(Counter(truth))
    et_count = dict(Counter(predict))

    # Calculate entropy for ground truth distribution
    for i, c in gt_count.items():
        gt_epy += (i * (c / n_key) * math.log2(n_key / c))

    # Calculate entropy for estimated distribution
    for i, c in et_count.items():
        et_epy += (i * (c / n_key) * math.log2(n_key / c))

    # Return absolute difference in entropies
    return abs(et_epy - gt_epy)