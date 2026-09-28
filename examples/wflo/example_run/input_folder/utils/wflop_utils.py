# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:21:41 2025

@author: matteobaricchi
"""

import numpy as np
from numpy import newaxis as na
from scipy.interpolate import RegularGridInterpolator


def filter_turbines(x,y,min_d):

    # ENFORCE TURBINE DISTANCE CONSTRAINT ---------------------------------
    
    # create distance matrix
    x_mat_1 = np.tile(np.reshape(x,(1,len(x))),(len(x),1))
    x_mat_2 = np.tile(np.reshape(x,(len(x),1)),(1,len(x)))
    y_mat_1 = np.tile(np.reshape(y,(1,len(y))),(len(y),1))
    y_mat_2 = np.tile(np.reshape(y,(len(y),1)),(1,len(y)))
    d_mat = np.sqrt((x_mat_1-x_mat_2)**2+(y_mat_1-y_mat_2)**2)

    # identify the turbines that have: d < min_d
    ind_mat = np.tile(np.reshape(np.arange(0,len(x)),(1,len(x))),(len(x),1))
    ind_mat[d_mat>=min_d] = -1
    ind_mat[np.triu(np.ones((len(x),len(x))),1)<1] = -1
    ind_turbine_delete_temp = ind_mat[ind_mat>=0]
    
    # create x,y new vectors with only relevant turbines
    ind_turbine_delete = np.unique(ind_turbine_delete_temp)
    ind_keep = np.delete(np.arange(0,len(x)),ind_turbine_delete)
    x_new = x[ind_keep]
    y_new = y[ind_keep]

    return x_new,y_new,ind_keep



def calculate_water_depth(x,y,site_x_grid,site_y_grid,site_bathymetry_grid):
    site_x_coord = site_x_grid[:,0]
    site_y_coord = site_y_grid[0,:]
    interp_function = RegularGridInterpolator((site_x_coord,site_y_coord),site_bathymetry_grid)
    water_depth = interp_function((x,y))    
    return water_depth



def randomize_layouts(x_mat_input,y_mat_input,boundaries,s_m,p_m):

    # check dimensions
    reconvert = False
    if x_mat_input.shape==y_mat_input.shape:
        if len(x_mat_input.shape)==1:
            x_mat_input = x_mat_input[na,:]
            y_mat_input = y_mat_input[na,:]
            reconvert = True
        elif len(x_mat_input.shape)>2:
            raise TypeError('Issue with dimensions')
    else:
        raise TypeError('Issue with dimensions')
    n_pop = x_mat_input.shape[0]
    n_wt = x_mat_input.shape[1]
            
    # create mutation bool matrix (0 = no mutation, 1 = mutation)
    mutation_bool_mat = np.random.rand(n_pop,n_wt)<p_m
    mutation_int_mat = np.zeros((n_pop,n_wt))
    mutation_int_mat[mutation_bool_mat] = 1
    
    # generate random step and direction
    step = s_m*np.random.rand(n_pop,n_wt)
    direction = ((2*np.pi))*np.random.rand(n_pop,n_wt)
    
    # apply mutation
    x_mat = x_mat_input+mutation_int_mat*step*np.cos(direction)
    y_mat = y_mat_input+mutation_int_mat*step*np.sin(direction)

    # enforce boundaries
    x_mat_output,y_mat_output = boundaries.enforce_boundaries_MultiPolygon(x_mat,y_mat)

    if reconvert:
        x_mat_output = x_mat_output.reshape(-1)
        y_mat_output = y_mat_output.reshape(-1)

    return x_mat_output,y_mat_output



def fix_turbines_positions(x,y,step_size,min_d,max_iter,boundary_obj):

    # initialization
    i = 0
    fix_needed = True

    while (i<max_iter) and (fix_needed):

        # create distance matrix
        x_mat_1 = np.tile(np.reshape(x,(1,len(x))),(len(x),1))
        x_mat_2 = np.tile(np.reshape(x,(len(x),1)),(1,len(x)))
        y_mat_1 = np.tile(np.reshape(y,(1,len(y))),(len(y),1))
        y_mat_2 = np.tile(np.reshape(y,(len(y),1)),(1,len(y)))
        d_mat = np.sqrt((x_mat_1-x_mat_2)**2+(y_mat_1-y_mat_2)**2)

        # identify the turbines that have: d < min_d
        ind_mat = np.tile(np.reshape(np.arange(0,len(x)),(1,len(x))),(len(x),1))
        ind_mat[d_mat>=min_d] = -1
        ind_mat[np.triu(np.ones((len(x),len(x))),1)<1] = -1
        ind_turbine_to_move = ind_mat[ind_mat>=0]
        
        # check if some turbines need to be moved
        fix_needed = len(ind_turbine_to_move)>0

        # move turbines
        if fix_needed:
            mutation_int_array = np.zeros(len(x))
            mutation_int_array[ind_turbine_to_move] = 1
            direction = ((2*np.pi))*np.random.rand(len(x))
            x_new_temp = x+(mutation_int_array*step_size*np.cos(direction))
            y_new_temp = y+(mutation_int_array*step_size*np.sin(direction))
            x_new_temp_mat,y_new_temp_mat = boundary_obj.enforce_boundaries_MultiPolygon(x_new_temp[na,:],y_new_temp[na,:])
            x_new = x_new_temp_mat.reshape(-1)
            y_new = y_new_temp_mat.reshape(-1)
            x = x_new.copy()
            y = y_new.copy()

        else:
            x_new = x.copy()
            y_new = y.copy()
            fix_needed = False

        #print(ind_turbine_to_move)
        i += 1
    
    if fix_needed:
        init_succeded = False
    else:
        init_succeded = True

    return init_succeded,x_new,y_new


def min_distance(x,y,diameter):
    x_mat_1 = np.tile(np.reshape(x,(len(x),1)),(1,len(x)))
    x_mat_2 = np.tile(np.reshape(x,(1,len(x))),(len(x),1))
    y_mat_1 = np.tile(np.reshape(y,(len(y),1)),(1,len(y)))
    y_mat_2 = np.tile(np.reshape(y,(1,len(y))),(len(y),1))
    d = np.sqrt((x_mat_1-x_mat_2)**2+(y_mat_1-y_mat_2)**2)
    np.fill_diagonal(d, np.inf)
    return np.min(d)/diameter
