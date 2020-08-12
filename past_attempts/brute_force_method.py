from itertools import combinations
from csv import DictReader
from scipy.stats import pearsonr
from math import sqrt



def read_file(f):
    
    reader = DictReader(open(f))
    
    data = {}
    for row in reader:
        
        for column, value in row.items():
            try:
                v = float(value)
            except ValueError:
                v = value
            data.setdefault(column, []).append(v)
    
    return data



def corr(X, Y):
    
    correlation = pearsonr(X, Y)[0]
    
    return correlation
    

    
def ranking(data, indeps, disp=False):   
    
    combos = []
    for length in range(3, len(indeps)+1):
        combo = combinations(indeps, length)
        
        for c in list(combo):
            combos.append(c)
    
    lengths = []
    combos_dict = {}
    
    for item in combos:
        sub_combo = combinations(item, 2)
        correlations = []
        lengths.append(len(item)) 

        
        for i in list(sub_combo):
            r = corr(data[i[0]], data[i[1]])
            correlations.append(r*r)
            
        combos_dict[item] = correlations
    
    ranks = {}
    for k,v in combos_dict.items():
        MSr = sqrt(sum(v))/len(v)
        weight = 1#len(k)/sum(lengths) 
        rankV = MSr/weight 
        ranks[k] = (rankV, MSr, weight)       
        
    ranks_sorted = sorted(ranks.items(), key=lambda x: x[1])

    if disp == True:
        for r in ranks_sorted[::-1]: 
            print('#{} Combo:'.format(ranks_sorted.index(r)+1), r[0])
            print('rankV = MSr / weight :')
            print(r[1])
            print()
    
    return ranks_sorted



def dep_correl(data, dep, excl):
    
    THRESH = .4 # Correlation threshold to response variable
    excl.append(dep)
    
    indep = []
    for d in data:
        if d not in excl:
            indep.append(d)
    
    for i in indep:
        r = corr(data[dep], data[i])
        
        if r != None:
            if abs(corr(data[dep], data[i])) < THRESH:
                indep.remove(i)

    return indep



DATA_FILE = 'mtcars.csv'
EXCLUDE = ['car', 'vs', 'am', 'gear', 'carb', 'disp'] # Keep as list
RESPONSE = 'mpg'

dta_main = read_file(DATA_FILE)
independent_vars = dep_correl(dta_main, RESPONSE, EXCLUDE)
regressor_combos = ranking(dta_main, independent_vars, disp=True)

RANK = 1 # Choose rank to print
#print(regressor_combos[RANK-1][0])