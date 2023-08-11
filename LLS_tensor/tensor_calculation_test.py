import yt
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle as pickle
from unyt import g,cm
from numpy.linalg import eig

box_A_root = '/data/gnedin/REI/D/Cai.B40.N256L2.sf=1_uv=0.15_bw=10_res=100.WC1/C/'
hc = pd.read_csv(box_A_root+'a=0.1453/hpropsRS.res', names=['Mvir','Mstar','Z','SFR','luminosity','px','py','pz','Rvir'],delim_whitespace=True,index_col=False)
ds = yt.load('/data/gnedin/REI/D/Cai.B40.N256L2.sf=1_uv=0.15_bw=10_res=100.WC1/C/rei40c1_a0.1453/rei40c1_a0.1453.art')
p = ds.artio_parameters
abox = p["abox"][0]
auni = p["auni"][0]
scale = auni/abox
##difference is llyman - the galaxies distances
data = []


def open_data():
    with open('peak_locations_box_c.pkl', 'rb') as f:
        peaks = pickle.load(f)
    return peaks
def vector():
    with open('vector.pkl', 'rb') as f:
        vector = pickle.load(f)
    return vector
