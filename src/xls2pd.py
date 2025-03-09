import pandas as pd

print('data/processed_lifespan_data.xlsx')
pld = pd.read_excel('data/processed_lifespan_data.xlsx', sheet_name=None) # {'Sheet1': <DataFrame> .shape=(4634,5) .columns: idx, set_genotype, weighted_mean, weighted_stddev, CV_score} 
print('data/lifespan_mccormick_data.xlsx')
lmd = pd.read_excel('data/lifespan_mccormick_data.xlsx', sheet_name=None) # {'rls': <DataFrame> .shape=(15094, 32) .columns: id, experiments, set_name, set_strain, ...}
print('data/expression_log_fold_changes.xlsx')
efc = pd.read_excel('data/expression_log_fold_changes.xlsx', sheet_name=None) # {'Sheet1': <DataFrame> .shape=(1484, 6184) .columns: idx, deleted_gene, gene activations x 6182.}
# print('data/interaction_strengths_of_yeastgenes.xlsx')
# isg = pd.read_excel('data/interaction_strengths_of_yeastgenes.xlsx', sheet_name=None)
print('data/41467_2023_43233_MOESM4_ESM.xlsx')
esm = pd.read_excel('data/41467_2023_43233_MOESM4_ESM.xlsx', sheet_name=None)
