import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from itertools import combinations



def z_distance(data, combo, dep):
    
    # Calculate Pearson correlation coefficient
    r = lambda x,y: pearsonr(x,y)[0] 
    # Apply Fisher's z transformation from |r| to z
    fisher =  lambda r: np.arctanh(np.abs(r))
    # Reverse transformation from z to r
    fisher_rev = lambda z: np.tanh(z)
    
    # Find every pairwise correlation within set
    r_indep = np.array([
        r(data[c[0]], data[c[1]]) for c in list(combinations(combo,2))
        ])
    
    # Find correlation with dependent variable for every variable in set
    r_dep = np.array([
        r(data[dep], data[var]) for var in combo
        ])
    
    # Apply Fisher's z transformation to all correlations
    r_indep = fisher(r_indep)
    r_dep = fisher(r_dep)
    
    # Calculate weighted average with weights determined by variables contribution to sum
    # Reverse transform back to correlation coefficient
    corr_in_indeps = fisher_rev(
        np.dot(r_indep, [r/sum(r_indep) for r in r_indep])
        )
    # Calculate weighted average with weights determined by variables contribution to sum
    # Reverse transform back to correlation coefficient        
    corr_to_dep = fisher_rev(
        np.dot(r_dep, [r/sum(r_dep) for r in r_dep])
        )

    # Calculate Euclidian distance from the point where:
    # correlation within independent variables is minimized (0)
    # correlation with dependent variable is maximized (1)
    distance = ( (0-corr_in_indeps)**2 + (1-corr_to_dep)**2 ) ** 0.5
    
    return distance



def total_combos(n_cols, minimum):
    
    # Define combination function ("n choose k")
    choose = lambda n,k: int(
        np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n-k))
        )
    
    # Sum up every number of combinations of every length
    return (
        sum(choose(n_cols, k) for k in np.arange(minimum, n_cols+1))
        )



class VariableChooser:
    
    def __init__(self, data, dep_var):

        self.data = data # Should be pandas DataFrame
        self.dep_var = dep_var
    
    def select_combo(self, indep_vars, minimum=2, len_penalty=False):
        
        assert len(indep_vars) >= 2
        assert minimum <= len(indep_vars)
           
        self.len_penalty = len_penalty 
        self.indep_vars = indep_vars
        self.min = minimum  
        distances = {}         
        
        # Create combinations of every length from minimum to number of indep vars
        for length in np.arange(self.min, len(self.indep_vars)+1):
            C = combinations(self.indep_vars, length)

            # Iterate over every column created
            for combo in list(C):  
                
                # Calculate 'z distance'
                distance = z_distance(self.data, combo, self.dep_var)
                
                # Add larger penalty to combinations of shorter length
                if len_penalty:
                    distance += 1 / (len(combo)*len(self.indep_vars))
                
                # Only add combo to memory if smaller than q1 of memory
                if len(distances) > 3:
                    if distance < np.percentile(list(distances.values()), 25):                  
                        distances[combo] = distance
                else:
                    distances[combo] = distance
                  
        # Sort combos by distance
        self.solutions = sorted(
            distances.items(),
            key = lambda x: x[1],
            )
        
        # Return combo with shortest distance
        return self.solutions[0]
    
    def total_combos(self):
        
        return total_combos(
            len(self.indep_vars), 
            self.min,
            )  
    
    def all_combos(self, indep_vars, minimum=2, len_penalty=True, lim=None):
        
        assert len(indep_vars) >= 2
        assert minimum <= len(indep_vars)
           
        # Nothing in this method will be saved to object memory
        data = self.data.copy(deep=True)
        dep_var = self.dep_var
        all_distances = {}         
        
        # Create combinations of every length from minimum to number of indep vars
        for length in np.arange(minimum, len(indep_vars)+1):
            C = combinations(indep_vars, length)

            # Iterate over every column created
            for combo in list(C):  
                
                # Calculate 'z distance'
                distance = z_distance(data, combo, dep_var)
                
                # Add larger penalty to combinations of shorter length
                if len_penalty:
                    distance += 1 / (len(combo)*len(indep_vars))

                all_distances[combo] = distance
                
        return list(all_distances.items())[:lim]


