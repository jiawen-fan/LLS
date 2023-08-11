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


def calculation(peaks):
    # sp = ds.sphere(ds.arr((peaks[2][0]/scale,peaks[2][1]/scale,peaks[2][2]/scale), 'Mpccm/h'), (.026/scale, "Mpccm/h"))
    I = []
    for i in peaks:
        sp = ds.sphere(ds.arr((i[0]/scale,i[1]/scale,i[2]/scale), 'Mpccm/h'), (0.026/scale, "Mpccm/h"))
    # sp = ds.sphere(ds.arr((peaks[0][0]/scale,peaks[0][1]/scale,peaks[0][2]/scale), 'Mpccm/h'), (0.026/scale, "Mpccm/h"))  
        x = sp[("gas","x")] - (sp.center.to('cm'))[0]
        y = sp[("gas","y")] - (sp.center.to('cm'))[1]
        z = sp[("gas","z")] - (sp.center.to('cm'))[2]
        volume = sp[('gas','dx')] *sp[('gas','dy')]*sp[('gas','dz')]
        ix_temp = np.multiply(sp[('gas','density')]*volume,(np.power(y,2) + np.power(z,2)))
        iy_temp = np.multiply(sp[('gas','density')]*volume,(np.power(x,2) + np.power(z,2)))
        iz_temp = np.multiply(sp[('gas','density')]*volume,(np.power(x,2) + np.power(y,2)))
        ixy_temp = -np.multiply(sp[('gas','density')]*volume,np.multiply(x,y))
        ixz_temp =  -np.multiply(sp[('gas','density')]*volume,np.multiply(x,z))
        iyz_temp = -np.multiply(sp[('gas','density')]*volume,np.multiply(z,y))
        Ix = np.sum(ix_temp)
        Iy = np.sum(iy_temp)
        Iz = np.sum(iz_temp)
        Ixy = np.sum(ixy_temp)
        Ixz = np.sum(ixz_temp)
        Iyz = np.sum(iyz_temp)
        I.append(np.array([[[Ix,Ixy,Ixz],[Ixy,Iy,Iyz],[Ixz,Iyz,Iz]]]))
    return(I)


def convertSphereToCart(theta, phi):
    "converts a unit vector in spherical to cartesian, needed for getGalaxies"
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

def eigen_stuff(I):
    eigen_vectors = []
    eigen_value = []
    eigen_value_total = []
    eigen_vector_total = []
    #max
    # eigen_vectors_max = 0
    # eigen_value_max = 0
    for i in I:
        print(i)
        eigenvalue, eigenvector= eig(i)
        #min
        eigen_vectors_max = eigenvector[0]
        eigen_value_max = eigenvalue[0]
        for i,z in zip(eigenvalue, eigenvector):
            #change this to largest or smallest
            if(i > eigen_value_max):
                eigen_value_max = i
                eigen_vectors_max = z
        eigen_value_total.append(eigenvalue)
        eigen_vector_total.append(eigenvector)
        eigen_vectors.append(eigen_vectors_max)
        eigen_value.append(eigen_value_max)
    return eigen_vectors,eigen_value, eigen_vector_total,eigen_value_total


def unit_vector(vector):
    return vector/np.linalg.norm(vector)

def angle_between(v1,v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)


# def _I_x(field,data):
#     #need to minus to get the right thing! from the original position
#     x = data[("gas","x")] 
#     y = data[("gas","y")] 
#     z = data[("gas","z")] 
#     return  np.multiply(data[('gas','density')],(np.power(y,2) + np.power(z,2)))
# ds.add_field(('gas','I_x'),_I_x,sampling_type = "cell",units = "g/cm",force_override=True)


# def _I_xy(field,data):
#     x = data[("gas","x")]
#     y = data[("gas","y")] 
#     z = data[("gas","z")] 
#     return  np.multiply(data[('gas','density')],np.multiply(x,y))
# ds.add_field(('gas','I_xy'),_I_xy,sampling_type = "cell",units = "g/cm",force_override=True)



def main():
    angle = []
    peaks = open_data()
    I = calculation(peaks)
    with open('locations_directions_box_c.pkl', 'rb') as f:
        locations = pickle.load(f)
    vector_total = []
    value_total = []
    vector_no_filter_total  = []
    value_no_filter_total =[]
    for i,z in zip(I,locations):
        vector,value,vector_total_,value_total_ = eigen_stuff(i)
        cartesian = convertSphereToCart(z[0],z[1])
        # angle.append(angle_between(cartesian,vector))
        vector_total.append(vector)
        value_total.append(value)
        vector_no_filter_total.append(vector_total_)
        value_no_filter_total.append(value_total_)
    # with open('angle.pkl', 'wb') as f:
    #     pickle.dump(angle, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('new_vector.pkl', 'wb') as f:
        pickle.dump(vector_total, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('new_value.pkl', 'wb') as f:
        pickle.dump(value_total, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('new_vector_total.pkl', 'wb') as f:
        pickle.dump(vector_no_filter_total, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('new_value_total.pkl', 'wb') as f:
        pickle.dump(value_no_filter_total, f, protocol=pickle.HIGHEST_PROTOCOL)
main()
