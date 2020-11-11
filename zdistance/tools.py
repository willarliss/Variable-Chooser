import sys
from collections import deque
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr



def z_distance(data, dep, mean='weighted'):
    
    assert mean in ['arithmetic', 'weighted', 'geometric', 'harmonic']
    
    # Calculate Pearson correlation coefficient
    corr = lambda x,y: pearsonr(x,y)[0] 
    
    # Define independent variables
    combo = np.array(data.drop(dep, axis=1).columns)
    
    # Find every pairwise correlation within set
    r_indep = np.array([
        corr(data[c[0]], data[c[1]]) for c in list(combinations(combo,2))
    ])
    
    # Find correlation with dependent variable for every variable in set
    r_dep = np.array([
        corr(data[dep], data[var]) for var in combo
    ])
    
    # Apply Fisher's z transformation to all correlations
    z_dep = np.arctanh(1-abs(r_dep))
    z_indep = np.arctanh(abs(r_indep))
    
    # Aggregate:
    if mean == 'arithmetic':
        corr_to_dep = sum(z_dep) / len(z_dep)
        corr_in_indeps = sum(z_indep) / len(z_indep)
    if mean == 'weighted':
        corr_to_dep = np.dot(z_dep, [z/sum(z_dep) for z in z_dep])        
        corr_in_indeps = np.dot(z_indep, [z/sum(z_indep) for z in z_indep])
    if mean == 'geometric':
        corr_to_dep = np.prod(z_dep) ** (1/len(z_dep))
        corr_in_indeps = np.prod(z_indep) ** (1/len(z_indep))
    if mean == 'harmonic':
        corr_to_dep = len(z_dep) / sum(1/z_dep)
        corr_in_indeps = len(z_indep) / sum(1/z_indep)
    
    # Calculate Euclidian distance from the point where:
    # correlation within independent variables is minimized (0)
    # 1 - correlation to dependent variable is minimized (0)
    distance = np.sqrt((0-corr_in_indeps)**2 + (0-corr_to_dep)**2) 
    
    return distance



def n_combos(n_cols, minimum):
    
    # Define combination function ("n choose k")
    choose = lambda n,k: int(
        np.math.factorial(n) / (np.math.factorial(k)*np.math.factorial(n-k))
    )
    
    # Sum up every number of combinations of every length
    return sum(choose(n_cols, k) for k in np.arange(minimum, n_cols+1))


 