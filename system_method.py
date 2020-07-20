from itertools import combinations
from scipy.stats import pearsonr
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math


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


def possible_combos(cols, dta):
    """Returns a list of each pairwise correlation"""
    
    combos_ = []
    C = combinations(cols, 2)
    
    for c in list(C):
        r = pearsonr(dta[c[0]], dta[c[1]])[0]
        combos_.append( (c[0],c[1],{'weight':round(r,5)}) )
    
    return combos_


def mean_sq(X):
    """Returns the square root of the mean of the squared correlations"""
    
    x = [edge[2]['weight']**2 for edge in X.edges(data=True)]
    avg = np.mean(x)
    return math.sqrt(avg)


def weight_comb(X):
    """Returns weighted means of the squared correlations"""

    x = np.array([edge[2]['weight']**2 for edge in X.edges(data=True)])
    weights = np.array([i/np.sum(x) for i in x])
    return weights.dot(x)


def system(indeps_, combos_, m, show=False):
    """Creates a graph with nodes as variables and edges as pair-wise correlation. 
    Determines the subgraph with the least mean squared correlation and returns a list
    of each subgraph with their corresponding mean squared correlation"""
    
    G = nx.Graph()
    G.add_nodes_from(indeps_)
    G.add_edges_from(combos_)
    
    nodes = set(G.nodes())    
    combs = []
    
    for length in range(m, len(nodes)+1):
        C = combinations(nodes, length)
        for c in list(C):
            combs.append(c)
    
    avgList = []
    for comb in combs:
        S = G.subgraph(list(comb))
        # avg = mean_sq(S)
        avg = weight_comb(S)
        avgList.append( (round(avg,5), comb) )
        
    avgList_s = sorted(avgList, key=lambda x: x[0])    

    if show:
        
        pos1 = nx.circular_layout(G)
        nx.draw(G, pos=pos1, with_labels=True)
        plt.show()
        
        for n in range(1, 4):
            plt.figure()
            S = G.subgraph(avgList_s[n][1])
            pos2 = nx.circular_layout(S)
            nx.draw(S, pos=pos2, with_labels=True)
            nx.draw_networkx_edge_labels(S, pos=pos2)       
            plt.show()
    
    return avgList_s


def main(dep, min_vars, dep_corr, f_name):
    """Given a set of variables, reports which combination of variables (of different 
    lengths) has the lowest correlation (measured by mean squared r)"""
    
    # Start with a set of variables
    data = read_file(f_name)
    
    # Exclude independent variables that are not correlated with dependent variable
    indeps = exclude(data, dep, dep_corr)
    
    # Define list of edges and weights (correlations) for system
    combos = possible_combos(indeps, data)
    
    # Build system
    min_corr_combos = system(indeps, combos, min_vars, show=False)
    
    # The lower the (root) mean squared correlation, the less average correlation
    [print(rank, i) for rank,i in zip(range(1,4), min_corr_combos[:3])] # top 3 combinations with lowest mean 


if __name__ == '__main__':
    
    DEP = 'mpg' # what is the dependent variable
    MIN_VARS = 4 # what is the minimum number of independent variables desired
    DEP_CORR = .45 # what is the minimum endogenous correlation desired
    F_NAME = 'mtcars.csv' # what is the file to be used
    
    main(DEP, MIN_VARS, DEP_CORR, F_NAME)   
    
