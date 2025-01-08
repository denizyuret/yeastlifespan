using XLSX, DataFrames

# print('data/processed_lifespan_data.xlsx')
# pld = pd.read_excel('data/processed_lifespan_data.xlsx')
# print('data/lifespan_mccormick_data.xlsx')
# lmd = pd.read_excel('data/lifespan_mccormick_data.xlsx')
# print('data/expression_log_fold_changes.xlsx')
# efc = pd.read_excel('data/expression_log_fold_changes.xlsx')
# print('data/interaction_strengths_of_yeastgenes.xlsx')
# isg = pd.read_excel('data/interaction_strengths_of_yeastgenes.xlsx')

@showtime pld = DataFrame(XLSX.readtable("data/processed_lifespan_data.xlsx", 1))
@showtime lmd = DataFrame(XLSX.readtable("data/lifespan_mccormick_data.xlsx", 1))
@showtime efc = DataFrame(XLSX.readtable("data/expression_log_fold_changes.xlsx", 1))
@showtime isg = DataFrame(XLSX.readtable("data/interaction_strengths_of_yeastgenes.xlsx", 1))
