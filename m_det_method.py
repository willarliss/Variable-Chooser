from itertools import combinations
from scipy.stats import pearsonr
import pandas as pd
import numpy as np


def read_file(f):
    """Read in .csv files and return as pandas dataframe. If file can't be read,
    empty pandas dataframe is returned"""
    
    try:
        df = pd.read_csv(f, delimiter=',')
    except Exception as e:
        print(e)
        df = pd.DataFrame([np.nan])
    return df


def exclude(dta, y, min_r):
    """Returns a list of variables that that are valid predictors according a specified 
    minimum level of correlation (min_r) with the dependent variable"""
    
    X_min_r = []
    X = [i for i in list(dta.columns) if i != y]
    
    for x in X:
        try:
            r = pearsonr(dta[x], dta[y])[0]
        except TypeError:
            r = None
        if r is not None and abs(r) > min_r:
            X_min_r.append(x)
    
    return X_min_r

        
def combo_dets(dta, indeps_, m):
    """Returns a list of all possible variables and from length k to the minimum length 
    specified (m). Each list item has a corresponding correlation matrix determinant value"""
    
    
    combs = []
    for length in range(m, len(indeps_)+1):
        C = combinations(indeps_, length)
        for c in list(C):
            combs.append(c)
    
    detList = []
    for comb in combs:
        sub = pd.DataFrame({i:dta[i] for i in comb})
        r_mat = np.array(sub.corr())
        r_det = np.linalg.det(r_mat)
        detList.append( (round(r_det,5),comb) )

    return sorted(detList, key=lambda x: x[0], reverse=True)    


def main(dep, min_vars, dep_corr, f_name):
    """Given a set of variables, reports which combination of variables (of different 
    lengths) has the lowest correlation (measured by correlation matrix determinant)"""

    # Start with a set of variables
    data = read_file(f_name)
    
    # Exclude independent variables that are not correlated with dependent variable
    indeps = exclude(data, dep, dep_corr)
    
    # Find determinants of correlation matrix for each subset
    min_corr_combos = combo_dets(data, indeps, min_vars)
    
    # The higher the correlation matrix determinant, the lower volume of correlation
    [print(rank, i) for rank,i in zip(range(1,4), min_corr_combos[:3])] # top 3 combinations with highest det

   
if __name__ == '__main__':
    
    DEP = 'mpg' # what is the dependent variable
    MIN_VARS = 4 # what is the minimum number of independent variables desired
    DEP_CORR = .45 # what is the minimum endogenous correlation desired
    F_NAME = 'mtcars.csv' # what is the file to be used
    
    main(DEP, MIN_VARS, DEP_CORR, F_NAME)
