import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def loss_rate(x):  # x is DataFrmae type
    table = ((x.isnull().sum())/x.shape[0]).sort_values(ascending=False).map(lambda i:"{:.5%}".format(i))
    return table