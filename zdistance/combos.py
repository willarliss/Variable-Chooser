import sys
from collections import deque
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from zdistance.tools import z_distance, n_combos



class VariableChooser:
    
    def __init__(self, data, dep_var, maxlen=10):

        self.data = data # Should be pandas DataFrame
        self.dep_var = dep_var
        self.maxlen = maxlen
        self.key = lambda x: x[1]
    
    def select_combo(self, indep_vars=None, maximum=None, minimum=3, len_penalty=False, agg='weighted'):
        
        if indep_vars is None:
            indep_vars = np.array(self.data.drop(self.dep_var, axis=1).columns)
        
        if maximum is None:
            maximum = len(indep_vars)
        assert maximum <= len(indep_vars)
        
        self.max = maximum
        self.min = minimum
        
        distances = deque(
            [('init', np.inf), ], 
            maxlen=self.maxlen,
        )

        # Create combinations of every length from minimum to number of indep vars
        for length in np.arange(self.min, self.max+1):
            C = combinations(indep_vars, length)

            # Iterate over every column created
            for combo in list(C):  
                
                # Calculate 'z-distance'
                distance = z_distance(
                    self.data[[*combo, self.dep_var]], 
                    dep=self.dep_var, 
                    mean=agg,
                )
                
                # Add larger penalty to combinations of shorter length
                if len_penalty:
                    distance += (len(indep_vars)-len(combo)+1) / (len(indep_vars)*len(combo))
                
                # Append to deque object if smaller than min item
                if distance < min(distances, key=self.key)[1]:
                    distances.append(
                        (combo, distance)
                    )
                  
        # Sort combos by distance and return combo with shortest distance
        self.solutions = sorted(distances, key=self.key)
        return self.solutions[0][0]
    
    def all_combos(self, indep_vars=None, maximum=None, minimum=3, len_penalty=False, agg='weighted', lim=None):
           
        if indep_vars is None:
            indep_vars = np.array(self.data.drop(self.dep_var, axis=1).columns)

        if maximum is None:
            maximum = len(indep_vars)
            
        self.max = maximum
        self.min = minimum
        
        all_distances = []      
        
        # Create combinations of every length from minimum to number of indep vars
        for length in np.arange(minimum, maximum+1):
            C = combinations(indep_vars, length)

            # Iterate over every column created
            for combo in list(C):  
                
                # Calculate 'z distance'
                distance = z_distance(
                    self.data[[*combo, self.dep_var]], 
                    dep=self.dep_var, 
                    mean=agg,
                )
                
                # Add larger penalty to combinations of shorter length
                if len_penalty:
                    distance += (len(indep_vars)-len(combo)+1) / (len(indep_vars)*len(combo))

                all_distances.append(
                    (combo, distance)
                )
                
        return sorted(all_distances, key=self.key)[:lim]

    def n_combos(self, maximum=None, minimum=None):
        
        if maximum is None:
            maximum = self.max
            
        if minimum is None:
            minimum = self.min
        
        return n_combos(
            maximum, 
            minimum,
        )  
      
      
       