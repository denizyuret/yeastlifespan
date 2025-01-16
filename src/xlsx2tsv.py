import pandas as pd
import sys
import os

infile = sys.argv[1]
outfile = os.path.splitext(infile)[0] + ".tsv"
df = pd.read_excel(infile)
df.to_csv(outfile, sep='\t', encoding='utf-8',  index=False)
