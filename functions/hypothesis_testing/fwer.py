"""Family-wise error rate corrections."""

import numpy as np


# 1) Bonferroni correction
# alpha = FWER threshold (family-wise error rate) i.e. the probability of making one or more false discoveries (type I errors) when performing multiple hypotheses tests.
def bonferroni_correction(p_values, alpha=0.5):
    m = len(p_values)
    adjusted_p_values = np.minimum(p_values * m, 1.0)
    significant = adjusted_p_values < alpha
    return adjusted_p_values, significant


# 2) Holm-Bonferroni correction
def holm_bonferroni_correction(p_values, alpha=0.5):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # The corrected value is put back into its original position, so the output corresponds to the input order.
    adjusted_p_values = np.zeros(m)
    largest_adjusted_p = 0.0
    
    for i in range(m):
        adjusted_p = min(sorted_p_values[i] * (m - i), 1.0)
        largest_adjusted_p = max(largest_adjusted_p, adjusted_p) # Simply checks that the next adj p-value is not smaller than the previous one, (from the m-k+1 denominator shrinking at greater rate as numerator growing).
        adjusted_p_values[sorted_indices[i]] = largest_adjusted_p
        
    significant = adjusted_p_values < alpha
    
    return adjusted_p_values, significant
