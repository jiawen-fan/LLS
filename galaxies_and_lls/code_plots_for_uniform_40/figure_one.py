import os
import numpy as np
from matplotlib import pyplot as plt      
import pandas as pd
from scipy import special                 
import array
import scipy as sp
import scipy.interpolate
import re
import pickle5 as pickle
from matplotlib.pyplot import figure
with open('40_non_uniform_lyman.pkl', 'rb') as f:
    lyman = pickle.load(f)
lyman_limited_value = []
for i in lyman:
        lyman_limited_value.append(i[0])
lyman_limited_value_sorted = sorted(lyman_limited_value)
min_value_lyman_limited_value = np.log10(min(lyman_limited_value_sorted))
max_value_lyman_limited_value = np.log10(max(lyman_limited_value_sorted))
bins = np.linspace(min_value_lyman_limited_value,max_value_lyman_limited_value,51)
bin_data = pd.DataFrame()
bin_data['cut_lyman_limited'] = np.array(lyman_limited_value_sorted)
bin_data['cut_lyman_limited'] = pd.cut(np.log10(bin_data['cut_lyman_limited']), bins = bins,include_lowest = True).astype(str)
number_count = []
for i in range(1, len(bins)):
    number_count.append(0)
    for k in range(0,len(lyman_limited_value_sorted)):
        if(np.log10(lyman_limited_value_sorted[k]) < bins[i]):
            number_count[i-1] += 1

number_count2 = []
for i in range(1,len(number_count)):
    number_count2.append(number_count[i] - number_count[i-1])

d_N_column= []
for i in range(1,50):
    d_N_column.append(10**bins[i]-10**bins[i-1])
dl = (200*1000)
y_value = []
for i,k in zip(number_count2,d_N_column):
    y_value.append(i/(dl*(k)))

x_value = []
for i in range(1,50):
    x_value.append(bins[i])

    

plt.yscale("log") 
y_axis = (np.multiply(np.power(10,x_value),y_value))
plt.plot(x_value,y_axis,color = "dimgray")
plt.savefig("f1_overplotting.png")