import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import combinations



class VariableChooser:
    
    def __init__(self, data, indep_vars, dep_var):

        assert len(indep_vars) >= 2

        self.data = data
        self.indep_vars = indep_vars
        self.dep_var = dep_var
        
        self.r = lambda x,y: pearsonr(x, y)[0]
        self.fisher = lambda r: np.arctanh(np.abs(r))
        self.fisher_rev = lambda z: (np.exp(2*z)-1) / (np.exp(2*z)+1)

    
    def correlations(self, minimum=2, method='weight', len_penalty=False):
        '''Aggregate correlations between each variable in a subset of independent 
        variables. Aggregate correlations between each independent variable and the
        dependent variable within a subset. Do this for every subset that exists as 
        a combination taken from a set of independent variables.'''
        
        if method == 'weight':
            self.weight, self.mean = True, False
        elif method == 'mean':
            self.weight, self.mean = False, True
        else:
            raise Exception('argument "how" must equal "weight" or "mean"')
        assert minimum <= len(self.indep_vars), \
            'minimum combo length may not be more than number of independ variables'

        ###
        
        def corr_to_dep(combo):
            '''Aggregate correlations between each independent variable and the 
            dependent variable. First calculate each correlation, then apply Fisher\'s
            Transformation, then use aggregation method (mean or weight). Apply penalty
            for subsets of shorter length if indicated.'''
            
            # Calculate the correlation of every sub-combination
            r_dep = np.array([
                self.r(self.data[self.dep_var], self.data[var]) for var in combo
                ]) 
            
            # Apply Fisher's z Transformation
            r_dep = self.fisher(r_dep)
            
            # Calculate weights based on each correlations contribution to the sum
            # Then take weighted average
            if self.weight:
                r = self.fisher_rev(
                    np.dot(r_dep, [i/sum(r_dep) for i in r_dep])
                    ) 
                
            # Calculate average
            if self.mean:
                r = self.fisher_rev(np.mean(r_dep))

            # Apply penalty equal to len(indep_vars)-len(combo)+1
            # Subtract r from 1 to turn maximization to minimization
            if len_penalty:
                p = r / (len(self.indep_vars)-len(combo)+1)
                return 1 - (r - p)
            # Subtract r from 1 to turn maximization to minimization
            else:
                return 1 - r   
            
        ###
        
        def corr_in_indeps(combo):
            '''Aggregate correlations between each variable is a subset of dependent
            variable. First calculate each correlation, then apply Fisher\'s Transformation, 
            then use aggregation method (mean or weight). Apply penalty for subsets of 
            shorter length if indicated.'''
              
            # Calculate the correlation of every sub-combination
            r_indep = np.array([ 
                self.r(self.data[c[0]], self.data[c[1]]) for c in list(combinations(combo,2)) 
                ])
            
            # Apply Fisher's z Transformation            
            r_indep = self.fisher(r_indep)
            
            # Calculate weights based on each correlations contribution to the sum
            # Then take weighted average            
            if self.weight:
                r = self.fisher_rev(
                    np.dot(r_indep, [i/sum(r_indep) for i in r_indep])
                    )
            
            # Calculate average
            if self.mean:
                r = self.fisher_rev(np.mean(r_indep)) 
                
            # Apply penalty equal to len(indep_vars)-len(combo)+1
            # Subtract r from 1 to turn maximization to minimization
            if len_penalty:
                p = r / (len(self.indep_vars)-len(combo)+1)
                return r - p
            # Subtract r from 1 to turn maximization to minimization
            else:
                return r
            
        ###
            
        self.len_penalty = len_penalty # Bool: apply penalty for shorter length or not
        self.min = minimum # Minimum length combination                
        self.indep_corr, self.dep_corr = {}, {}
        
        # Create combinations of every possible length from the set of variables
        for length in np.arange(self.min, len(self.indep_vars)+1):

            # C object creates combinations from given list of given length
            C = combinations(self.indep_vars, length)

            # Iterate over every combination produced by C
            for combo in list(C):
                r_dep = corr_to_dep(combo)
                r_indep = corr_in_indeps(combo)
                
                # Save aggregated correlation to respective dictionaries
                self.dep_corr[combo] = r_dep
                self.indep_corr[combo] = r_indep
                            
        return self.indep_corr, self.dep_corr   
    
    def plot(self, tol=0.5):
        '''Plot 1 minus the intercorrelation to the dependent variable against the
        intercorrelation within the independent variables. Define a circle of
        "tolerance" as a fixed distance from the origin.'''
        
        assert 0 <= tol <= 1
        
        # Extract values of inter-correlation within the independent variables
        indep_corr = np.array(
            list(self.indep_corr.values())
            )
        
        # Extract values of inter-correlation within to the dependent variable
        dep_corr = np.array(
            list(self.dep_corr.values())
            )
                          
        # Plot:
        fig, ax = plt.subplots(figsize=(7,6))
        ax.plot(indep_corr, dep_corr, 'b.', alpha=0.5)   
        # Add "tolerance circle"          
        ax.add_patch(plt.Circle([0,0], tol, color='r', alpha=0.5))
        
        # Customizes axes
        ax.set_xlim([0,1]) 
        ax.set_ylim([0,1])
        ax.set_xlabel('inter-correlation within independent variables')
        ax.set_ylabel('1 - inter-correlation to dependent variable')
        
    def solve(self):
        '''Determine which subset of variables minimizes the inter-correlation
        within the independent variables and maximizes the intercorrelation to the
        dependent variable.'''
        
        distances = {}
        
        # Iterate every combination that was previously generated
        for combo in self.indep_corr.keys():
            
            # Pull out the individual point values for each combination
            x = self.indep_corr[combo]
            y = self.dep_corr[combo]
            
            # Calculate each point's distance from the origin and add to dictionary
            distance = ( (0-x)**2 + (0-y)**2 ) ** 0.5
            distances[combo] = distance

        # Sort the dictionary from least to greatest
        distances = sorted(
            distances.items(),
            key = lambda x: x[1],
            )
        
        # Return dictionary with rankings included
        return list(
            zip(np.arange(1, len(distances)+1), distances)
            )
    
    def total_combos(self, n_cols=None, minimum=None):
        '''Calculate the total number of combinations that can exist within a set
        of a given length and a minimum length. Sum_i(n C k_i) where n is the total 
        length of a given set and k_i is an iterable between that and the minimum.'''
        
        # If no columns number is specified, use previously fit data
        if n_cols is None:
            n_cols = len(self.indep_vars)
        # If no minimum is specified, use previously defined minimum
        if minimum is None:
            minimum = self.min
        
        # Define combination function ("n choose k")
        choose = lambda n,k: int(
            np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n-k))
            )
        
        # Sum up every number of combinations of every length
        self.n_combos = (
            sum(choose(n_cols, k) for k in np.arange(minimum, n_cols+1))
            )
        
        return self.n_combos
  
    
    
def test():    
    
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
        print(solution[i]) # rank, combo, distance
        print('-'*22)
        print(df[combo].corr()) # correl of combination
        print(1-vc.dep_corr[tuple(combo)]) # Agg corr to dep var
        print()
    
    vc.plot(tol=0.689)
    
    
  
if __name__ == '__main__':
    
    test()


