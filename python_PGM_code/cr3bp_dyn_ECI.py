# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:48:25 2025

@author: tarun
"""

import numpy as np
from numpy import linalg as la
import scipy as sci
from scipy import io as sio
import datetime as dt
import pymap3d
import cr3bp_dyn as cr3bp

def termSat(T, Y):
    mu = 1.2150582e-2 # Dimensionless mass of the moon (and position of Earth w.r.t. barycenter)
    Rm = 1740/384400 # Nondimensionalized radius of the moon
    value = (np.sqrt((Y[0] + mu)**2 + Y[1]**2 + Y[2]**2) < 6371/384400) or (np.sqrt((Y[0] - (1-mu))**2 + Y[1]**2 + Y[2]**2) < Rm) # Stop when the target hits the Earth's or the Moon's surface
    return 1

save_path = "D:\\PythonProjects\\EDP\\PGM_Git\\PAR-PGM\\"
# Define initial conditions
mu = 1.2150582e-2
# x0 = [0.5-mu, 0.0455, 0, -0.5, 0.5, 0.0]' # Sample Starting Point
# x0 = [-1.00506 0 0 -0.5 0.5 0.0]' # L3 Lagrange Point
# x0 = [0.5-mu, sqrt(3/4), 0, 0, 0, 0]' # L4 Lagrange Point

# Lagrange points that we may not use
# x0 = [0.836915 0 0 0 0 0]' # L1 Lagrange Point
# x0 = [1.15568 0 0 0 0 0]' # L2 Lagrange Point
# x0 = [0.5-mu, -sqrt(3/4), 0, 0, 0, 0]' # L5 Lagrange Point

# x0 = [-0.144158380406153	-0.000697738382717277	0	0.0100115754530300	-3.45931892135987	0] # Planar Mirror Orbit "Loop-Dee-Loop" Sub-Trajectory
# x0 = [1.15568 0 0 0 0.04 0]'
x0 = np.array([1.16429257222878, -0.0144369085836121, 0, -0.0389308426824481, 0.0153488211249537, 0]) # L2 Lagrange Point Approach

# Coordinate system conversions
dist2km = 384400 # Kilometers per non-dimensionalized distance
time2hr = 4.342*24 # Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60) # Kms per non-dimensionalized velocity

# Define time span
tstamp1 = 0 # For long term trajectories 
# tstamp = 0.3570
end_t = 36/time2hr - tstamp1
tspan = np.arange(0, end_t, 6.25e-3) # For our modified trajectory 

# Call ode45()

# [t,dx_dt] = ode45(@cr3bp_dyn, tspan, x0, opts); # Assumes termination event (i.e. target enters LEO)

dx_dt = np.zeros((tspan.shape[0], x0.shape[0]))
dx_dt[0,:] = x0
t = np.zeros((tspan.shape[0],1))
termSat.terminal = True
termSat.direction = 0
for i in range(tspan.shape[0]-1):
    ivp_res = sci.integrate.solve_ivp(cr3bp.cr3bp_dyn, (0, tspan[i+1]-tspan[i]), np.copy(x0), method='BDF', events=termSat, rtol=1e-6, atol=1e-8)
    x0 = ivp_res.y.T[-1,:]
    dx_dt[i+1,:] = x0[:]
    t[i+1] = tspan[i+1]

# tstamp = 40*24/time2hr;

# Longer-term scheduling
tstamp = t[-1] # Begin new trajectory where we left off
end_t = (40*24)/time2hr
tspan = np.arange(tstamp[0], end_t, 8/time2hr) # Schedule to take measurements once every 8 hours
x0_tmp = np.zeros(dx_dt[-1,:].shape)
x0_tmp[:] = dx_dt[-1,:]
t = t[:-1].reshape(-1)
dx_dt = dx_dt[:-1,:]

dx_dts = np.zeros((tspan.shape[0], x0.shape[0]))
dx_dts[0,:] = x0_tmp[:] # Start at end of pass
ts = np.zeros((tspan.shape[0],1))
ts[0] = tstamp
termSat.terminal = True
termSat.direction = 0
for i in range(tspan.shape[0]-1):
    ivp_res = sci.integrate.solve_ivp(cr3bp.cr3bp_dyn, (tspan[i], tspan[i+1]), np.copy(x0_tmp), method='BDF', events=termSat, rtol=1e-6, atol=1e-8)
    x0_tmp = ivp_res.y.T[-1,:]
    dx_dts[i+1,:] = x0_tmp[:]
    ts[i+1] = ivp_res.t[-1]

t = np.hstack((t, ts.reshape(-1)))
dx_dt = np.vstack((dx_dt, dx_dts))

rb = dx_dt[:,:3] # Position evolutions from barycenter
vb = dx_dt[:,3:] # Velocity evolutions from barycenter
rbe = np.array([[-mu], [0], [0]]) # Position vector relating center of earth to barycenter

# Insert code for obtaining vector between center of Earth and observer

obs_lat = 30.618963
obs_lon = -96.339214
elevation = 103.8

# dtLCL = datetime('now', 'TimeZone','local'); # Current Local Time
# dtUTC = datetime(dtLCL, 'TimeZone','Z');     # Current UTC Time
# UTC_vec = datevec(dtUTC); # Convert to vector

UTC_vec = dt.datetime(2024, 5, 3, 2, 41, 15, tzinfo=dt.timezone.utc)
t_add_dim = tstamp1 * (4.342)
UTC_vec = UTC_vec + dt.timedelta(t_add_dim)

# reo_dim = lla2eci([obs_lat obs_lon elevation], UTC_vec); # Position vector between observer and center of Earth in meters
# reo_nondim = reo_dim/(1000*384400);

rem = np.array([[1], [0], [0]]) # Earth center - moon center 

reo_nondim = np.zeros((t.shape[0],3))
veo_nondim = np.zeros((t.shape[0],3)) # Array for non-dimensionalized EO velocity vectors

rot = np.zeros(rb.shape) # Observer - Target 
rom = np.zeros(rb.shape) # Observer - Moon Center
vot = np.zeros(rb.shape) # Observer - Target Velocity

for i in range(rb.shape[0]):
    t_add_nondim = t[i] - tstamp1 # Time since first point of orbit
    t_add_dim = t_add_nondim * (4.342) # Conversion to dimensionalized time
    delt_add_dim = t_add_dim - 1/86400 

    updated_UTCtime = UTC_vec + dt.timedelta(t_add_dim)
    
    delt_updatedUTCtime = UTC_vec + dt.timedelta(delt_add_dim)
    
    reo_dim = pymap3d.geodetic2eci(obs_lat, obs_lon, elevation, updated_UTCtime)
    delt_reodim = pymap3d.geodetic2eci(obs_lat, obs_lon, elevation, delt_updatedUTCtime)
    reo_dim = np.asarray(reo_dim).reshape(3)
    delt_reodim = np.asarray(delt_reodim).reshape(3)
    veo_dim = reo_dim - delt_reodim
    
    R_z = np.array([[np.cos(t_add_nondim + tstamp1), -np.sin(t_add_nondim + tstamp1), 0],
                    [np.sin(t_add_nondim + tstamp1), np.cos(t_add_nondim + tstamp1), 0],
                    [0, 0, 1]])
    
    dRz_dt = np.array([[-np.sin(t_add_nondim + tstamp1), -np.cos(t_add_nondim + tstamp1), 0], 
                       [np.cos(t_add_nondim + tstamp1), -np.sin(t_add_nondim + tstamp1), 0], 
                       [0, 0, 0]])
    
    reo_nondim[i,:] = reo_dim[:]/(1000*384400) # Conversion to non-dimensional units and ECI frame
    veo_nondim[i,:] = veo_dim[:]*(4.342*86400)/(1000*384400)
    
    rot[i,:] = -reo_nondim[i,:] + (R_z@(-rbe + rb[i,:].reshape(rbe.shape))).reshape(-1)
    rom[i,:] = -reo_nondim[i,:] + (R_z@rem).reshape(-1)
    vot[i,:] = -veo_nondim[i,:] + (R_z@(vb[i,:].reshape(rbe.shape))).reshape(-1) + (dRz_dt@(-rbe + rb[i,:].reshape(rbe.shape))).reshape(-1)

# Before we obtain AZ and EL quantities, we must convert our
# observer-target vector into a topocentric frame.

rot_topo = np.zeros((t.shape[0],rot.shape[1]))
rom_topo = np.zeros((t.shape[0],rot.shape[1]))
vot_topo = np.zeros((t.shape[0],rot.shape[1]))

for i in range(t.shape[0]):
    # Step 1: Find the unit vectors governing this topocentric frame
    z_hat_topo = reo_nondim[i,:]/la.norm(reo_nondim[i,:])

    x_hat_topo_unorm = np.cross(z_hat_topo, np.array([0, 0, 1])) # We choose a reference vector 
    x_hat_topo = x_hat_topo_unorm/la.norm(x_hat_topo_unorm) # Remember to normalize
    # such as the North Pole, we have several choices regarding this
    y_hat_topo_unorm = np.cross(x_hat_topo, z_hat_topo)
    y_hat_topo = y_hat_topo_unorm/la.norm(y_hat_topo_unorm) # Remember to normalize

    # Step 2: Convert all of the components of 'rot' from our aligned reference
    # frames to this new topocentric frame.

    rot_topo[i,:] = np.array([np.dot(rot[i,:], x_hat_topo), np.dot(rot[i,:], y_hat_topo), np.dot(rot[i,:], z_hat_topo)])
    rom_topo[i,:] = np.array([np.dot(rom[i,:], x_hat_topo), np.dot(rom[i,:], y_hat_topo), np.dot(rom[i,:], z_hat_topo)])

    # Step 3: Handle the time derivatives of vot_topo = d/dt (rot_topo)
    R_topo = np.vstack((x_hat_topo, y_hat_topo, z_hat_topo)) # DCM relating ECI to topocentric coordinate frame
    dmag_dt = np.dot(reo_nondim[i,:], veo_nondim[i,:])/la.norm(reo_nondim[i,:]) # How the magnitude of r_eo changes w.r.t. time
    
    zhat_dot_topo = (veo_nondim[i,:]*la.norm(reo_nondim[i,:]) - reo_nondim[i,:]*dmag_dt)/(la.norm(reo_nondim[i,:]))**2;
    xhat_dot_topo = np.cross(zhat_dot_topo, np.array([0, 0, 1]))/la.norm(np.cross(z_hat_topo, np.array([0, 0, 1]))) - np.dot(x_hat_topo, np.cross(zhat_dot_topo, np.array([0, 0, 1])))*x_hat_topo
    yhat_dot_topo = (np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))/la.norm(np.cross(x_hat_topo, z_hat_topo)) - np.dot(y_hat_topo, np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))*y_hat_topo

    dA_dt = np.vstack((xhat_dot_topo, yhat_dot_topo, zhat_dot_topo))

    vot_topo[i,:] = (R_topo@vot[i,:].reshape((-1, 1)) + dA_dt@rot[i,:].reshape((-1, 1))).reshape(-1)

# Due to not being able to see targets behindthe moon, design function such 
# that if the rot_topo vector passes through the Moon, then data for that 
# time step is considered invalid.

rot_valid = []
vot_valid = []
t_valid = []

j = -1
Rm = 1740/384400 # Nondimensionalized radius of the moon

for i in range(t.shape[0]):
    if (la.norm(np.cross(rot_topo[i,:], rom_topo[i,:]))/la.norm(rot_topo[i,:]) > Rm and (t[i] <= tstamp or t[i] > (30*24)/time2hr)):
        j += 1
        t_valid.append(t[i])
        rot_valid.append(rot_topo[i,:])
        vot_valid.append(vot_topo[i,:])
    

rot_valid = np.asarray(rot_valid)
vot_valid = np.asarray(vot_valid)
t_valid = np.asarray(t_valid)

# Convert observer - target position vectors into range, azimuth, and
# elevation quantities

Rho = np.zeros((rot_valid.shape[0],1))
AZ = np.zeros((rot_valid.shape[0],1))
EL = np.zeros((rot_valid.shape[0],1))

for i in range(rot_valid.shape[0]):
    Rho[i,0] = np.sqrt(rot_valid[i,0]**2 + rot_valid[i,1]**2 + rot_valid[i,2]**2)
    AZ[i,0] = np.arctan2(rot_valid[i,1], rot_valid[i,0])
    EL[i,0] = np.pi/2 - np.arccos(rot_valid[i,2]/Rho[i,0])

# Last Step: Due to elevation angle constraints, all t, Rho, AZ, EL data
# for which EL < 0 is considered invalid and should be discarded
t_valid = t_valid.reshape([-1, 1])
full_ts = np.hstack((t_valid.reshape([-1, 1]), Rho, AZ, EL)) # Full augmented time-series vector
first_msmt = np.where(full_ts[:,0] > 6)[0][0]

valid_El = np.where(full_ts[:,3] > 0)[0]
valid_Az = np.where(np.abs(full_ts[:,2]) < 0.5*np.pi)[0]
partial_ts_ECI = full_ts[valid_El, :]


np.savetxt(save_path + "partial_ts.csv", partial_ts_ECI, delimiter=',', fmt='%f')
np.savetxt(save_path + "full_ts.csv", np.hstack((t.reshape([-1, 1]), rot_topo)), delimiter=',', fmt='%f')
np.savetxt(save_path + "full_vts.csv", np.hstack((t.reshape([-1, 1]), vot_topo)), delimiter=',', fmt='%f')

