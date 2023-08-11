import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.interpolate as interp
import scipy as sp
import re
def unPackRawFile(raw_path):
    """
    - unpacks the Raw conditions file. Not used for the neural network.
    All data in real space.
    """
    y = np.loadtxt(raw_path, skiprows=6)
    distance = y[:,0]
    pec_vel = y[:,1]
    temp = y[:,2]
    HI_density = y[:,3]
    #gas_density = y[:,4]
    #gas_metallicity = y[:,5]
    
    return distance, pec_vel, temp, HI_density#, gas_density, gas_metallicity

def unPackRawFlux(flux_path):
    """
    - unpacks the .fHI. Not used for the neural network. In redshift space.
    """
    y2 = np.genfromtxt(flux_path, skip_header=1, delimiter=' , ')
    velocity = y2[:,0]
    flux = y2[:,1] 
    return velocity, flux

def hubble_flow_convert(velocity, a, omega_m, omega_lam):
    """
    - uses hubble flow to convert from velocity to distance
    """
    aH = a * 100 * (omega_m / a ** 3 + omega_lam)** 0.5
    return velocity/aH

def resample(distance, item, new_distance):
    """
    - interpolates the distances so that we can resample. useful because the velocity after converting using hubble flow doesn't have the same positions as the underlying properties.
    - creates a consistent distance scale (obviously these distances are messed up by peculiar velocities)
    """
    f = interp.interp1d(distance, item)
    new_item = f(new_distance)
    
    return new_item


def getDir(path_LOS,linenumber=5):
    """
    the direction of the LOS is given inside each file, (in the comments)
    this function parses the comments to get that information
    """
    f = open(path_LOS)
    x = f.readlines()[linenumber]
    answer = re.search('\(([^)]+)', x.split(', ')[1]).group(1)
    arr = np.array(answer.split(','),dtype=float)
    return arr

def getPos(path_LOS,linenumber=5):
    """
    the start position of the LOS is given inside each file, (in the comments)
    this function parses the comments to get that information
    """
    f = open(path_LOS)
    x = f.readlines()[linenumber]
    answer = re.search('\(([^)]+)', x).group(1)
    arr = np.array(answer.split(','),dtype=float)
    
    return arr

def convertSphereToCart(theta, phi):
    """
    converts a unit vector in spherical to cartesian, needed for getGalaxies"
    """
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    
    
def get_galaxies(gals, path_LOS, number):
    """
    function can take in the path of the galaxy file (.res) and the line of sight. Number should match the LOS #
    """   
    # Parameter
    Lbox = 80  # box size
    aexp = 0.1453  # scale factor for the epoch
    OmegaM = 0.3036
    h100 = 0.6814    # normalized Hubble constant
    pos = np.array(getPos(path_LOS))  # origin of LOS
    pos = pos/512*Lbox
    sphere_los = np.array(getDir(path_LOS))  # direction of LOS , np.sum(e3**2) should be = 1
    e3 = convertSphereToCart(sphere_los[0], sphere_los[1])

    halo = gals[0] # SFR
    xg = gals[1] #\
    yg = gals[2] # | positions of the galaxies in cMpc/h
    zg = gals[3] #/
    vx = gals[4] #\
    vy = gals[5] # | velocities of the galaxies in km/s
    vz = gals[6] #/
    
    out_arr = Box(Lbox, OmegaM, h100, aexp, halo, xg, yg, zg, vx, vy, vz, pos, e3)
    num_arr = number*np.ones(len(out_arr.T))
    return np.vstack([out_arr, num_arr])

def Box(Lbox, OmegaM, h100, aexp, halo, xg, yg, zg, vx, vy, vz, pos, e3):
    """
    Wraps LOS to the original box (0-40), helper function for get galaxies.
    """
    ##### find intersection between LOS and grid
    planeNormal = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    if e3[0] > 0:
        planePoint_x = np.array([[80, 0, 0],[160, 0, 0],[240, 0, 0]])
        planePoint_x_inverse = np.array([0,0,0])
    else:
        planePoint_x = np.array([[0, 0, 0],[-80, 0, 0],[-120, 0, 0]])
        planePoint_x_inverse = np.array([80,0,0])
    if e3[1] > 0:
        planePoint_y = np.array([[0, 80, 0],[0, 160, 0],[0, 240, 0]])
        planePoint_y_inverse = np.array([0,0,0])
    else:
        planePoint_y = np.array([[0, 0, 0],[0, -80, 0],[0, -160, 0]])
        planePoint_y_inverse = np.array([0,80,0])
    if e3[2] > 0:
        planePoint_z = np.array([[0, 0, 80],[0, 0, 160],[0, 0, 240]])
        planePoint_z_inverse = np.array([0,0,0])
    else:
        planePoint_z = np.array([[0, 0, 0],[0, 0, -80],[0, 0, -160]])
        planePoint_z_inverse = np.array([0,0,80])
    
    planePoint = np.vstack((planePoint_x,planePoint_y,planePoint_z))
    intercept_pts = LinePlaneIntersection(planeNormal=planeNormal, planePoint=planePoint, losDirection=e3, losPoint=pos)
    intercept_pts[:,0][0:3] = planePoint_x[:,0]
    intercept_pts[:,1][3:6] = planePoint_y[:,1]
    intercept_pts[:,2][6:] = planePoint_z[:,2]
    planePoint_inverse = np.vstack((planePoint_x_inverse,planePoint_y_inverse,planePoint_z_inverse))
    intercept_pts_inverse = LinePlaneIntersection(planeNormal=planeNormal, planePoint=planePoint_inverse, losDirection=(-1)*e3, losPoint=pos)
    for i in range(3):
        intercept_pts_inverse[i,i] = planePoint_inverse[i,i]

    los_start_pts_inverse = np.where(np.logical_and(e3>0, intercept_pts_inverse == 0), 80, intercept_pts_inverse)
    distance_inverse = (-1)*np.sqrt(np.sum((intercept_pts_inverse-pos)**2,axis=1))
    distance = np.sqrt(np.sum((intercept_pts-pos)**2,axis=1))
    new_los_start_pts = np.vstack((pos,intercept_pts[distance<270]%80))
    new_los_start_pts = np.where(np.logical_and(e3<0, new_los_start_pts == 0), 80, new_los_start_pts)
    new_los_start_pts = np.vstack((new_los_start_pts,los_start_pts_inverse))
    x = xg
    y = yg
    z = zg

    new_x = np.tile(x, (len(new_los_start_pts), 1))
    new_y = np.tile(y, (len(new_los_start_pts), 1))
    new_z = np.tile(z, (len(new_los_start_pts), 1))


    w = e3[0]*(new_x-new_los_start_pts[:,0][:, np.newaxis])+\
        e3[1]*(new_y-new_los_start_pts[:,1][:, np.newaxis])+\
        e3[2]*(new_z-new_los_start_pts[:,2][:, np.newaxis])

    dx = new_x - new_los_start_pts[:,0][:, np.newaxis] - w*e3[0]
    dy = new_y - new_los_start_pts[:,1][:, np.newaxis] - w*e3[1]
    dz = new_z - new_los_start_pts[:,2][:, np.newaxis] - w*e3[2]
    new_x = np.where(dx>40,new_x-80,new_x)
    new_x = np.where(dx<-40,new_x+80,new_x)
    new_y = np.where(dy>40,new_y-80,new_y)
    new_y = np.where(dy<-40,new_y+80,new_y)
    new_z = np.where(dz>40,new_z-80,new_z)
    new_z = np.where(dz<-40,new_z+80,new_z)

    w = e3[0]*(new_x-new_los_start_pts[:,0][:, np.newaxis])+\
        e3[1]*(new_y-new_los_start_pts[:,1][:, np.newaxis])+\
        e3[2]*(new_z-new_los_start_pts[:,2][:, np.newaxis])

    dx = new_x - new_los_start_pts[:,0][:, np.newaxis] - w*e3[0]
    dy = new_y - new_los_start_pts[:,1][:, np.newaxis] - w*e3[1]
    dz = new_z - new_los_start_pts[:,2][:, np.newaxis] - w*e3[2]

    distance_all = np.concatenate((np.insert(distance[distance<270],0,0),distance_inverse))[:, np.newaxis]
    w += distance_all

    # distance from the galaxy to the LOS
    dr = np.sqrt(dx**2+dy**2+dz**2)

    # convert w from real to redshift space
    aH = 100*aexp*np.sqrt(OmegaM/aexp**3+(1-OmegaM))
    w += (e3[0]*vx+e3[1]*vy+e3[2]*vz)/aH

    # convert dr from comoving Mpc/h to proper Mpc
    dr *= (aexp/h100)
    halo = np.tile(halo, (len(new_los_start_pts), 1))

    # select galaxies

    sel1 = np.logical_and(w>5,w<195) # within first 100 cMpc/h
    sel2 = np.logical_and(dr<2,halo>0.0001)
    sel = np.logical_and(sel1,sel2)

    xpos_g = np.tile(xg, (len(new_los_start_pts), 1))
    ypos_g = np.tile(yg, (len(new_los_start_pts), 1))
    zpos_g = np.tile(zg, (len(new_los_start_pts), 1))
    
    if(len(sel) > 0):
        halo1 = halo[sel]
        w1 = w[sel]
        w1 = np.round(w1,decimals=8)
        dr1 = dr[sel]
        dr1 = np.round(dr1,decimals=8)
        xpos_g1 = xpos_g[sel]
        xpos_g1 = np.round(xpos_g1,decimals=8)
        ypos_g1 = ypos_g[sel]
        ypos_g1 = np.round(ypos_g1,decimals=8)
        zpos_g1 = zpos_g[sel]
        zpos_g1 = np.round(zpos_g1,decimals=8)
        #gives halo, distance along LOS, and distance away from LOS
        return np.unique(np.array([halo1,w1,dr1,xpos_g1,ypos_g1,zpos_g1]),axis=1)
    else:
        return np.array([[],[],[],[],[],[]])

def LinePlaneIntersection(planeNormal, planePoint, losDirection, losPoint):
    """
    Finds the intersection between LOS and grid
    This function does a lot of floating point floating point arithmetic
    so the intersection at x=0 may be at x=-1e16, as an example
    I correct the result in Box() because intersection at 0 means the LOS starting pt is 40
    """
    ndotu = planeNormal.dot(losDirection)
    w = losPoint - planePoint
    si = np.array([-planeNormal[i].dot(w[i*len(w)//3:(i+1)*len(w)//3].T) / ndotu[i] for i in [0,1,2]])
    Psi = w + np.vstack(np.multiply.outer(si,losDirection)) + planePoint
    return Psi


def get_density_data(path_ifrit_file):
    lint = np.int64
    with open(path_ifrit_file,'r') as f:
        h = np.fromfile(f,dtype=lint,count=1); assert(h[0] == 12)
        nn = np.fromfile(f,dtype=np.int32,count=3)
        h = np.fromfile(f,dtype=lint,count=1); assert(h[0] == 12)
        n = nn[0]
        assert(n==nn[1] and n==nn[2])
        x = np.fromfile(f,dtype=lint,count=1);
        assert(len(x) == 1)
        assert(x[0] == 4*n**3)
        den = np.fromfile(f,dtype=np.float32,count=n**3)
        x = np.fromfile(f,dtype=lint,count=1);
        assert(x[0] == 4*n**3)
        x = np.fromfile(f,dtype=lint,count=1);
        assert(len(x) == 1)
        assert(x[0] == 4*n**3)
        tem = np.fromfile(f,dtype=np.float32,count=n**3)
        x = np.fromfile(f,dtype=lint,count=1);
        assert(x[0] == 4*n**3)
        den = den.reshape((n,n,n))
        tem = tem.reshape((n,n,n))

    return den,tem

def ReadMesh(file,recl=8):
    """
    Reads in xHI data, use recl=8 for both 40Mpc and 80Mpc data
    """
    if(recl == 4):
        lint = np.int32
    else:
        assert(recl == 8)
        lint = np.int64
    with open(file,'r') as f:
        h = np.fromfile(f,dtype=lint,count=1); assert(h[0] == 12)
        nn = np.fromfile(f,dtype=np.int32,count=3)
        h = np.fromfile(f,dtype=lint,count=1); assert(h[0] == 12)
        n = nn[0]
        assert(n==nn[1] and n==nn[2])

        ret = []

        while(1):
            x = np.fromfile(f,dtype=lint,count=1);
            if(len(x) == 0): return ret
            assert(x[0] == 4*n**3)
            d = np.fromfile(f,dtype=np.float32,count=n**3).reshape((n,n,n))
            x = np.fromfile(f,dtype=lint,count=1); assert(x[0] == 4*n**3)
            ret.append(d)

def get_density_LOS(density,path_LOS):
    pos = getPos(path_LOS)  # origin of LOS
    pos = pos/256*1024  # to match ifrit file
    sphere_los = np.array(getDir(path_LOS))  # direction of LOS , np.sum(e3**2) should be = 1
    e3 = convertSphereToCart(sphere_los[0], sphere_los[1])
    sample_pts = np.linspace(20,2540,2520*3)
    dx = sample_pts*e3[0]
    dy = sample_pts*e3[1]
    dz = sample_pts*e3[2]
    sample_xpos = np.where((pos[0]+dx)<0,pos[0]+dx+1024*3,pos[0]+dx)%1024
    sample_ypos = np.where((pos[1]+dy)<0,pos[1]+dy+1024*3,pos[1]+dy)%1024
    sample_zpos = np.where((pos[2]+dz)<0,pos[2]+dz+1024*3,pos[2]+dz)%1024

    return density[np.floor(sample_zpos).astype(int),np.floor(sample_ypos).astype(int),np.floor(sample_xpos).astype(int)]

def get_LOS_distance_zspace(path_LOS, aexp, OmegaM, new_distance):
    """
    Distance along LOS is in real space. Needs to use peculiar velocity in .raw file to convert to redshift space.
    new_distance is in real space (where data is sampled)
    """
    distance, pec_vel, _, _ = unPackRawFile(path_LOS)
    aH = 100*aexp*np.sqrt(OmegaM/aexp**3+(1-OmegaM))
    x_redshift_space = distance + pec_vel/aH
    f = interp.interp1d(distance, x_redshift_space)
    new_distance_redshift_space = f(new_distance)
    
    return new_distance_redshift_space



def main_function():
    gals = np.loadtxt('hpropsRS.res', usecols=[8,5,6,7],unpack=1)
    gals_vel = np.zeros((3,gals.shape[1]))
    gals = np.concatenate((gals,gals_vel))
    los_gal_list = []
    for i in np.arange(0,30):
            #pulling data from galaxies
        flux_path ='los.00' + '{0:03}'.format(i) +'.raw'
        los_gal_data = get_galaxies(gals, flux_path, i)
        los_gal_list.append(los_gal_data)
    return los_gal_list

gal_list = main_function()
import pickle

with open('galaxies_location.pkl','wb') as f:
    pickle.dump(gal_list,f)