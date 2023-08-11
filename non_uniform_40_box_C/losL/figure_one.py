import os
import numpy as np
from matplotlib import pyplot as plt      
import pandas as pd
from scipy import special                 
import array
import scipy as sp
import scipy.interpolate
import re
import pickle as pickle
from matplotlib.pyplot import figure
with open('color_data0_cul_integral.pkl', 'rb') as f:
    lyman = pickle.load(f)
with open('color_data1_cul_integral.pkl', 'rb') as f:
    lyman1= pickle.load(f)
lyman_limited_value = []
for i in lyman:
        lyman_limited_value.append(i[0][0][-1])
for i in lyman1:
        lyman_limited_value.append(i[0][0][-1])
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

with plt.style.context('science','notebook'):
    plt.figure()    
    plt.plot(x_value,y_value,color = "dimgray")
    plt.rcParams['font.size'] = 8
    plt.yscale("log") 
    plt.yticks()
    plt.xlabel("$\mathrm{log10(N_{HI})}$",fontsize = 10)
    plt.ylabel("$\mathrm{dN/(dl\: dN_{HI})}$",fontsize =10,)
    plt.savefig("f1_cul.pdf")



y_axis = (np.multiply(np.power(10,x_value),y_value))


with plt.style.context('science','notebook'):
    plt.figure()    
    plt.plot(x_value,y_axis,color = "dimgray")
    plt.rcParams['font.size'] = 8
    plt.yscale("log") 
    plt.yticks()
    plt.xlabel("$\mathrm{log10(N_{HI})}$",fontsize = 10)
    plt.ylabel("$\mathrm{N_{HI}}$ $\mathrm{dN/(dl\: dN_{HI})}$",fontsize =10,)
    plt.savefig("f1_overplotting_cul.pdf")
 
#observational data
column_density_spacing = [10**11,10**14,10**16,10**18,10**20,10**21.5,10**22]
x_axis_2= [11.0,14.0000,16,18,20,21.5,22]
median = [10**-8.4,10**-12.3,10**-15.87,10**-18.94,10**-21.51,10**-23.41,10**-25.08]
mean = [10**-8.38,10**-12.30,10**-16.00,10**-19.12,10**-21.20,10**-23.60,10**-24.94]
percentile_16th  = [10**(-8.38-.5),10**(-12.3-0.07),10**(-16.0-0.14),10**(-19.12-0.11),10**(-21.2-0.16),10**(-23.60-0.13),10**(-24.94-0.46)]
percentile_84th  = [10**(-8.38+.5),10**(-12.3+0.07),10**(-16.0+0.14),10**(-19.12+0.11),10**(-21.2+0.16),10**(-23.60+0.13),10**(-24.94+0.46)]
stuff = np.multiply(column_density_spacing,mean)
upper_bounds = np.multiply(np.subtract(percentile_84th, mean),column_density_spacing)
lower_bounds =  np.subtract(np.multiply(mean,column_density_spacing),np.multiply(percentile_16th,column_density_spacing))
scale_factor = (1+5.8)**2 *68/3e5
lower_bound = []
for i in lower_bounds:
    lower_bound.append(float(i))
upper_bound = []
for i in upper_bounds:
    upper_bound.append(float(i))
error = [lower_bound, upper_bound]
median_scaled = np.multiply(median,scale_factor)
error_scaled = np.multiply(error,scale_factor)

with plt.style.context('science','notebook'):
    plt.figure()    
    plt.plot(x_value,y_axis,color = "dimgray")
    plt.rcParams['font.size'] = 8 
    plt.errorbar(x_axis_2, np.multiply(median_scaled,column_density_spacing),yerr = error_scaled,fmt="o",ms=2)
    plt.yscale("log")
    plt.yticks()
    plt.xlabel("$\mathrm{log10(N_{HI})}$",fontsize = 10)
    plt.ylabel("$\mathrm{N_{HI}}$ $\mathrm{dN/(dl\: dN_{HI})}$",fontsize =10)
    plt.savefig("simple_fit_cul.pdf")
