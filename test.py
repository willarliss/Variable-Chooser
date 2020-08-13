import sys
import numpy as np
import pandas as pd

from z_distance_combos import VariableChooser, z_distance, total_combos

def main():
    
    data = pd.read_csv('mtcars.csv')
    cols = list(col for col in data.columns if col not in ['car', 'mpg'])
            
    vc = VariableChooser(data, 'mpg')

    solution = vc.select_combo(cols, minimum=3, len_penalty=True)
    
    for i in np.arange(10):
        print(vc.solutions[i])
        
if __name__ == '__main__':
    main()
    
