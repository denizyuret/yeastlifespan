# XLSX buggy, opened https://github.com/felipenoris/XLSX.jl/issues/278
# using XLSX, DataFrames, OrderedCollections

# function read_all_tabs(xlsx_file)
#     XLSX.openxlsx(xlsx_file) do xf
#         OrderedDict(n => DataFrame(XLSX.gettable(xf[n])) for n in XLSX.sheetnames(xf))
#     end
# end

# Using python instead
using OrderedCollections, DataFrames
import Pandas, XLSX
sheetnames(f)=XLSX.sheetnames(XLSX.readxlsx(f))
read_all_tabs(f)=(n=sheetnames(f); length(n) == 1 ? DataFrame(Pandas.read_excel(f)) : OrderedDict(i=>DataFrame(Pandas.read_excel(f,i)) for i in n))

@showtime pld = read_all_tabs("data/processed_lifespan_data.xlsx")
@showtime lmd = read_all_tabs("data/lifespan_mccormick_data.xlsx")
@showtime efc = read_all_tabs("data/expression_log_fold_changes.xlsx")
#@showtime isg = read_all_tabs("data/interaction_strengths_of_yeastgenes.xlsx")
@showtime esm = read_all_tabs("data/41467_2023_43233_MOESM4_ESM.xlsx")
