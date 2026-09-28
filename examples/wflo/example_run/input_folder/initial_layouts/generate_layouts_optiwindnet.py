#%%
import numpy as np
import pickle
import utm
import time

from optiwindnet.augmentation import poisson_disc_filler


# extract HKN data
with open(f'../HKN_data_and_tools/HKN_data.pkl', 'rb') as f:
    HKN_data = pickle.load(f)
hkn_boundaries_x = HKN_data['hkn_boundaries_x']
hkn_boundaries_y = HKN_data['hkn_boundaries_y']
hkn_wt_x = HKN_data['hkn_wt_x']
hkn_wt_y = HKN_data['hkn_wt_y']
diameter = 283.2
n_wt = len(hkn_wt_x)

# scale HKN data - boundaries and initial positions
coord_sub = utm.from_latlon(52.70,4.29)
x_sub = coord_sub[0]
y_sub = coord_sub[1]
diameter_hkn = 200.
hkn_wt_x_scaled = x_sub + (hkn_wt_x-x_sub)*(diameter/diameter_hkn)
hkn_wt_y_scaled = y_sub + (hkn_wt_y-y_sub)*(diameter/diameter_hkn)
hkn_boundaries_x_scaled = x_sub + (hkn_boundaries_x-x_sub)*(diameter/diameter_hkn)
hkn_boundaries_y_scaled = y_sub + (hkn_boundaries_y-y_sub)*(diameter/diameter_hkn)

# define population number and samples
n_pop = 350 
n_samples = np.array([1,2,3,4,5,6,7,8,9,10],dtype=int)

for n in n_samples:

    t = time.time()
    min_d = 4*diameter
    hkn_boundaries_xy_scaled = np.vstack((hkn_boundaries_x_scaled,hkn_boundaries_y_scaled)).T
    seeds = np.arange(1,n_pop+1)+n_pop*(n-1)
    x_mat_initial = np.zeros((n_pop,n_wt))
    y_mat_initial = np.zeros((n_pop,n_wt))

    for i in np.arange(len(seeds)):
        coord = poisson_disc_filler(n_wt,min_dist=min_d,BorderC=hkn_boundaries_xy_scaled,seed=seeds[i])
        x_mat_initial[i,:] = coord[:, 0]
        y_mat_initial[i,:] = coord[:, 1]

    # store data
    data = {}
    data['x_mat_initial'] = x_mat_initial
    data['y_mat_initial'] = y_mat_initial
    with open(f'initial_pop_v{n}.pkl', 'wb') as f:
        pickle.dump(data, f)

    print(f'Generation layout {n} of {len(n_samples)} completed - Time: {time.time()-t}')
    
