import patsy
import time
import sys
import pandas as pd
import numpy as np

import covbat as cb

# import importlib
# importlib.reload(cb)

# read data from R output
pheno = pd.read_table('bladder-pheno.txt', index_col=0)
dat = pd.read_table('bladder-expr.txt', index_col=0)

mod = patsy.dmatrix("~ age + cancer", pheno, return_type="dataframe")

#### CovBat test ####
# record time
t = time.time()
ebat = cb.covbat(dat, pheno['batch'], mod, "age")
sys.stdout.write("%.2f seconds\n" % (time.time() - t))

sys.stdout.write(str(ebat.iloc[:5, :5]))

ebat.to_csv("py-covbat.txt", sep="\t") # save Python output
p = pd.read_table('py-covbat.txt', index_col=0)
r = pd.read_table('r-covbat.txt', index_col=0)

assert (p - r).max().max() < 1e-4

#### ComBat test ####t = time.time()
t = time.time()
ebat = cb.combat(dat, pheno['batch'], mod, "age")
sys.stdout.write("%.2f seconds\n" % (time.time() - t))

sys.stdout.write(str(ebat.iloc[:5, :5]))

ebat.to_csv("py-combat.txt", sep="\t") # save Python output
p = pd.read_table('py-combat.txt', index_col=0)
r = pd.read_table('r-combat.txt', index_col=0)

assert (p - r).max().max() < 1e-4
