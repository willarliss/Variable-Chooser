import pandas as pd
from sklearn.datasets import load_boston
from z_distance_combos import VariableChooser, z_distance, total_combos

boston = load_boston()
df = pd.DataFrame(boston.data, columns=[list(boston.feature_names)])
df.columns = [col[0] for col in df.columns.values]
df['PRICE'] = boston.target

print(z_distance(df, ['INDUS', 'NOX', 'AGE', 'TAX', 'RAD'], 'PRICE'))
print(z_distance(df, ['DIS', 'RM', 'PTRATIO', 'LSTAT', 'TAX'], 'PRICE'))

vc = VariableChooser(df, 'PRICE')

cols = df.drop('PRICE', axis=1).columns
solution = vc.select_combo(cols, minimum=3, len_penalty=True)
print(solution)

for i in np.arange(10):
    print(vc.solutions[i])
    
