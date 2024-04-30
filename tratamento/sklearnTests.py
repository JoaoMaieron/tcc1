import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Retirar os três primeiros dias de cada (começa em 03/01/2010)



# path= './csv_inmet/final_diario/GO.csv'
# data = pd.read_csv(path, parse_dates=['data'])

path = './sales/train.csv'
data = pd.read_csv(path, index_col='id', parse_dates=['date'])
a=6