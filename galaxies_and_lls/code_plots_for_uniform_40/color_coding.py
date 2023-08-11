import pickle5 as pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import special                 
import array
import scipy as sp
from scipy import spatial
from scipy.stats import binned_statistic
import re
def distance(points1, points2):
    return(np.sqrt((points1[0]-points2[0])**2 +(points1[1]-points2[1])**2+(points1[2]-points2[2])**2))
def nfvir(lyman_distance,n):
    gals = np.loadtxt('hpropsRS.res', usecols=[5,6,7,8],unpack=1) 
    points = []
    for i,j,k in zip(gals[0],gals[1],gals[2]):
        points.append([i,j,k])
    radius_points = []
    for i in gals[3]:
        radius_points.append(i)
    #lls in rvir
    #total lls
    #bins count
    #one extra space
    points.append([0,0,0])
    lymans = []
    for i in (lyman_distance):
        #i[0] is the column density, i[1] is the lyman location
        #less than certain density in bins

        #check everything inside a 4 level ball, instead of constructing multiple
        
        points[-1] = (i[1])
        tree = spatial.KDTree(points)
        #look at what is associated
        all_nn_indices = tree.query_ball_point(points[-1],r=4.979e-01*4) 
        condition = 5
        for z in range(n,0,-1):
        #check fvir conditions
            for k in all_nn_indices:
                if(points[k] != points[-1]):
                #see if distance in between is smaller than vir distance times by n 
                #add mass constraints
                    if(distance(i[1],points[k]) < radius_points[k]*z):
                        condition = z
                        #if true
        lymans.append([i,condition])
    return lymans
test = []
with open('40_non_uniform_lyman.pkl', 'rb') as f:
    lyman_limited_value = pickle.load(f)

for i in range(0,10):
    lymans= nfvir(lyman_limited_value[i],4)
    with open('color_data' + i + '.pkl', 'wb') as f:
        pickle.dump(lymans, f, protocol=pickle.HIGHEST_PROTOCOL)



