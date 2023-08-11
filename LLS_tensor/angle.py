import yt
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle as pickle
from unyt import g,cm
from numpy.linalg import eig
#I took the largest eigen, instead the min eigen, difference between angle 2 and 3 
with open('new_value.pkl','rb')as f:
    value = pickle.load(f)
with open("new_vector.pkl",'rb') as f:
    vector = pickle.load(f)
# with open('lyman_cul.pkl', 'rb') as f: 
#     lyman  = pickle.load(f) 
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
    plot_var.append(i[0])

with open('angle_new_max.pkl', 'wb') as f:
    pickle.dump(plot_var, f, protocol=pickle.HIGHEST_PROTOCOL)
plt.hist(plot_var)
plt.xlabel("cos(theta)")
plt.ylabel("number counts")
plt.savefig('histogram7.png') 