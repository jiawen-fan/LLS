import yt
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle as pickle
from unyt import g,cm
from numpy.linalg import eig

with open('value_total.pkl','rb')as f:
    value_list = pickle.load(f)
with open("vector_total.pkl",'rb') as f:
    vector_list = pickle.load(f)
# with open('lyman_cul.pkl', 'rb') as f: 
#     lyman  = pickle.load(f) 
vector = []
for i in range(len(vector_list)):

    min = value_list[i][0][0]
    index = 0
    for z in range(3):
        #change here to change if you want to look at min eigen value or max
        if value_list[i][0][z] <min:
            min = value_list[i][0][z]
            index = z
    vector.append(vector_list[i][0][index])
for i in range(10):
    print(vector[i])

def unit_vector(vector):
    return vector/np.linalg.norm(vector)

def convertSphereToCart(theta, phi):
    "converts a unit vector in spherical to cartesian, needed for getGalaxies"
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
def angle_between(v1,v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)
with open('locations_directions_box_c.pkl', 'rb') as f:
    locations = pickle.load(f)
angle = []
for i,z in zip(vector,locations):
    cartesian = convertSphereToCart(z[0],z[1])
    angle.append(abs(angle_between(i,cartesian)))

plot_var = []
for i in angle:
    plot_var.append(i)

with open('angle_min_eigen.pkl', 'wb') as f:
    pickle.dump(plot_var, f, protocol=pickle.HIGHEST_PROTOCOL)
plt.hist(plot_var)
plt.savefig('histogram5.png') 