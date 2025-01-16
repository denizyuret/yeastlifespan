import pandas as pd

print('data/processed_lifespan_data.xlsx')
pld = pd.read_excel('data/processed_lifespan_data.xlsx', sheet_name=None)
print('data/lifespan_mccormick_data.xlsx')
lmd = pd.read_excel('data/lifespan_mccormick_data.xlsx', sheet_name=None)
print('data/expression_log_fold_changes.xlsx')
efc = pd.read_excel('data/expression_log_fold_changes.xlsx', sheet_name=None)
# print('data/interaction_strengths_of_yeastgenes.xlsx')
# isg = pd.read_excel('data/interaction_strengths_of_yeastgenes.xlsx', sheet_name=None)
print('data/41467_2023_43233_MOESM4_ESM.xlsx')
esm = pd.read_excel('data/41467_2023_43233_MOESM4_ESM.xlsx', sheet_name=None)
