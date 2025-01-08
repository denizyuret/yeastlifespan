import pandas as pd

print('data/processed_lifespan_data.xlsx')
pld = pd.read_excel('data/processed_lifespan_data.xlsx')
print('data/lifespan_mccormick_data.xlsx')
lmd = pd.read_excel('data/lifespan_mccormick_data.xlsx')
print('data/expression_log_fold_changes.xlsx')
efc = pd.read_excel('data/expression_log_fold_changes.xlsx')
print('data/interaction_strengths_of_yeastgenes.xlsx')
isg = pd.read_excel('data/interaction_strengths_of_yeastgenes.xlsx')
