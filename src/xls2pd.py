import pandas as pd

print('data/processed_lifespan_data.xlsx')
pld = pd.read_excel('data/processed_lifespan_data.xlsx') # {'Sheet1': <DataFrame> .shape=(4634,5) .columns: idx, set_genotype, weighted_mean, weighted_stddev, CV_score} 
print('data/lifespan_mccormick_data.xlsx')
lmd = pd.read_excel('data/lifespan_mccormick_data.xlsx') # {'rls': <DataFrame> .shape=(15094, 32) .columns: id, experiments, set_name, set_strain, ...}
print('data/expression_log_fold_changes.xlsx')
efc = pd.read_excel('data/expression_log_fold_changes.xlsx') # {'Sheet1': <DataFrame> .shape=(1484, 6184) .columns: idx, deleted_gene, gene activations x 6182.}
# print('data/interaction_strengths_of_yeastgenes.xlsx')
# isg = pd.read_excel('data/interaction_strengths_of_yeastgenes.xlsx') # {'cc_NN': <DataFrame> .shape=(4784,4785) .columns: Unnamed: 0 Unnamed: 1  YMR056C  YBR085W  YJR155W  YNL331C  YOL165C  YFL057C  YCR107W  ...  YMR243C  YNR039C  YER033C  YGL255W  YLR130C  YKL175W  YBR046C  YGR285C  YNL241C}
# Using sheet_name=None to load all sheets, default is to load only the first sheet. The previous xls files only have one sheet.
print('data/41467_2023_43233_MOESM4_ESM.xlsx')
esm = pd.read_excel('data/41467_2023_43233_MOESM4_ESM.xlsx', sheet_name=None)
