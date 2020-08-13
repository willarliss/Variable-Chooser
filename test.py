import os
import sys
import numpy as np
import pandas as pd
from distance_z_method import VariableChooser



def main():    
    
    # Generate synthetic data
    np.random.seed(3) 
    df = pd.DataFrame(np.random.randint(1,9, 100), columns=['a'])
    df['b'] = df['a'] + np.random.randn(100)
    df['c'] = df['b'] + np.random.randn(100)
    df['d'] = df['c'] + np.random.randn(100)
    df['e'] = df['d'] + np.random.randn(100)
    df['f'] = df['e'] + np.random.randn(100)
    df['g'] = df['f'] + np.random.randn(100)
    df['h'] = df['g'] + np.random.randn(100)
    df.drop('a', axis=1, inplace=True)
    
    vc = VariableChooser(df, ['b','c','d','f','g','h'], 'e')
    
    indep, dep = vc.correlations(method='weight', minimum=2, len_penalty=True)
    solution = vc.solve()
    
    n = 5
    for i in np.arange(n):
        combo = list(solution[i][1][0])    
        print(solution[i])
        print('-'*22)
        print(df[combo].corr())
        print(1-vc.dep_corr[tuple(combo)])
        print()
    
    vc.plot(tol=0.689)
    
    
    
if __name__ == '__main__':
    
    main()
    
    
    