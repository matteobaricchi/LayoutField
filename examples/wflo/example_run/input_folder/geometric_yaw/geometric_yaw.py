# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:47:16 2025

@author: matteobaricchi
"""

import numpy as np


def calculate_dxdy(x,y,k,wd,diameter):
    
    # extend dimensions of x and y
    x_mat_1 = np.tile(np.reshape(x,(len(x),1)),(1,len(x)))
    x_mat_2 = np.tile(np.reshape(x,(1,len(x))),(len(x),1))
    y_mat_1 = np.tile(np.reshape(y,(len(y),1)),(1,len(y)))
    y_mat_2 = np.tile(np.reshape(y,(1,len(y))),(len(y),1))

    # calculate dx and dy
    d = np.sqrt((x_mat_1-x_mat_2)**2+(y_mat_1-y_mat_2)**2)
    theta = np.arctan2(y_mat_2-y_mat_1,x_mat_2-x_mat_1)
    gamma = wd*(np.pi/180)-(3/2)*np.pi+theta
    dx = d*np.cos(gamma)
    dy = d*np.sin(gamma)
    
    # identify waked turbines
    #condition_waked = np.logical_and(dx>0, np.abs(dy)<=(diameter/2+k*dx))
    condition_waked = np.logical_and(dx>0, np.abs(dy)-diameter/2<=(diameter/2+k*dx))

    # calculate number of waked turbine
    n_t_waked = np.sum(condition_waked,1)
    
    # assign +inf to dx and dy of unwaked turbines
    dx_waked = np.ones(dx.shape)*np.inf
    dy_waked = np.ones(dy.shape)*np.inf
    dx_waked[condition_waked] = dx[condition_waked]
    dy_waked[condition_waked] = dy[condition_waked]
    
    
    # extarct dx,dy of the closest turbine (filter neighbour)
    d_waked = np.sqrt(dx_waked**2+dy_waked**2)
    wt_neigh = np.argmin(d_waked,1)
    wt_neigh_mat = np.tile(np.reshape(wt_neigh,(len(wt_neigh),1)),(1,len(wt_neigh)))
    wt_count_mat = np.tile(np.reshape(np.arange(0,len(wt_neigh)),(1,len(wt_neigh))),(len(wt_neigh),1))
    fil_neigh = wt_neigh_mat == wt_count_mat
    dx_neigh = dx[fil_neigh]
    dy_neigh = dy[fil_neigh]


    # filter values (only turbines whose wake affects other turbines)
    fil_wake = n_t_waked>0
    n_t_waked_fil = n_t_waked[fil_wake]
    dx_neigh_fil = dx_neigh[fil_wake]
    dy_neigh_fil = dy_neigh[fil_wake]
    dx_all_fil = np.reshape(dx_waked[np.tile(np.reshape(fil_wake,(len(fil_wake),1)),(1,len(x)))],(len(dx_neigh_fil),len(x)))
    dy_all_fil = np.reshape(dy_waked[np.tile(np.reshape(fil_wake,(len(fil_wake),1)),(1,len(y)))],(len(dy_neigh_fil),len(y)))
    
    #return dx,dy,condition_waked
    return fil_wake,dx_neigh_fil,dy_neigh_fil,n_t_waked_fil,dx_all_fil,dy_all_fil






def calculate_geomYaw_ExpCorr(x,
                              y,
                              wd,
                              ws,
                              ws_rated,
                              wind_turbine,
                              wfm,
                              yaw_max = 21.846,             # coefficient tuned for IEA22MW
                              p_x = 4.889,                  # coefficient tuned for IEA22MW
                              p_y = 9.594,                  # coefficient tuned for IEA22MW
                              q_x = 5.820,                  # coefficient tuned for IEA22MW
                              q_y = 0.380,                  # coefficient tuned for IEA22MW
                              alpha_f_ws_eff = 0.150,       # coefficient tuned for IEA22MW
                              w_corr = 0.456                # coefficient tuned for IEA22MW
                              ):


    # extract diameter from the turbine
    diameter = wind_turbine.diameter()
    
    # set entraintment constant
    k = 0.1
    
    # initialize geometric yaw array
    yaw_array = np.zeros((len(x),len(wd),len(ws)))
    
    # calculate effective wind speed for all turbines, wind directions and wind speeds
    ws_eff_ilk = wfm(x,y,wd=wd,ws=ws,tilt=0,yaw=0).WS_eff_ilk
    
    # iterate for each wind direction
    for wd_ind in np.arange(0,len(wd)):
    
        # calculate dx and dy for the nearest waked turbine for one wind direction
        fil_wake,dx_neigh,dy_neigh,_,dx_all,dy_all = calculate_dxdy(x,y,k,wd[wd_ind],diameter) 
        dx_all_ext = np.tile(np.reshape(dx_all,(len(dx_neigh),len(x),1)),(1,1,len(ws)))
        dy_all_ext = np.tile(np.reshape(dy_all,(len(dy_neigh),len(y),1)),(1,1,len(ws)))
        
        # associate the effective wind speed of the nearest waked turbine each turbine
        ws_eff = ws_eff_ilk[:,wd_ind,:]
        ws_eff_mat = np.tile(np.reshape(ws_eff,(1,len(x),len(ws))),(len(x),1,1))
        ws_eff_mat_fil = ws_eff_mat[fil_wake,:,:]
        fil_ws_eff = np.ones((len(dx_neigh),len(x),len(ws)))
        
        # filter out turbines such that: ws_eff>ws_rated
        fil_ws_eff[ws_eff_mat_fil>ws_rated] = 0
        
        # filter out turbines such that: ws_eff<<ws_cut_in
        ws_cut_in = 4
        delta_cut_in = 2 # NEED PROPER TUNING
        fil_ws_eff[ws_eff_mat_fil<ws_cut_in-delta_cut_in] = 0
        
        # find the nearest turbine in the wake (after filtering)
        dx_all_ext[fil_ws_eff<1] = np.inf
        dy_all_ext[fil_ws_eff<1] = np.inf
        
        # calculate f_ws_eff
        #alpha_f_ws_eff = 0.3 # NEED PROPER TUNING
        f_ws_eff = 1-alpha_f_ws_eff*np.exp(np.reshape(ws_eff[fil_wake,:],(len(dx_neigh),1,len(ws)))-np.tile(np.reshape(ws,(1,1,len(ws))),(len(dx_neigh),1,1)))
        
        
        # initialize geometric yaw angles for one wind direction
        yaw_temp = np.zeros((len(x),1,len(ws)))
        eps = 1e-10
        
        
        # calculate the first approx of geometric angle
        
        # distinguish dy>0 and dy<0
        dy_all_ext_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_2 = -np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dx_all_ext_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dx_all_ext_2 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_1[dy_all_ext>=0] = dy_all_ext[dy_all_ext>=0]
        dy_all_ext_2[dy_all_ext<0] = dy_all_ext[dy_all_ext<0]
        dx_all_ext_1[dy_all_ext>=0] = dx_all_ext[dy_all_ext>=0]
        dx_all_ext_2[dy_all_ext<0] = dx_all_ext[dy_all_ext<0]
        
        # calculate first approx for both dy>0 and dy<0
        yaw_geom_mat_1 = np.sign(dy_all_ext_1+eps)*yaw_max*((1+p_x)/(p_x+np.e**(dx_all_ext/(q_x*diameter))))*((1+p_y)/(p_y+np.e**(np.abs(dy_all_ext_1)/(q_y*diameter))))
        yaw_geom_mat_2 = np.sign(dy_all_ext_2+eps)*yaw_max*((1+p_x)/(p_x+np.e**(dx_all_ext/(q_x*diameter))))*((1+p_y)/(p_y+np.e**(np.abs(dy_all_ext_2)/(q_y*diameter))))
        yaw_geom_first_approx_1 = np.amax(yaw_geom_mat_1,axis=1)
        yaw_geom_first_approx_2 = np.amin(yaw_geom_mat_2,axis=1)
        
        # choose dominant influence (dy>0 or dy<0) and select the first approximation
        yaw_geom_first_approx = yaw_geom_first_approx_1
        yaw_geom_first_approx[yaw_geom_first_approx_1<np.abs(yaw_geom_first_approx_2)] = yaw_geom_first_approx_2[yaw_geom_first_approx_1<np.abs(yaw_geom_first_approx_2)]
        
        
        # extract dx and dy relevant for both dy>0 and dy<0
        
        ind_1 = np.argmax(yaw_geom_mat_1,axis=1,keepdims=True)
        ind_2 = np.argmin(yaw_geom_mat_2,axis=1,keepdims=True)
        fil_ind_1 = np.tile(ind_1,(1,len(x),1))==np.tile(np.reshape(np.arange(len(x)),(1,len(x),1)),(len(dx_neigh),1,len(ws)))
        fil_ind_2 = np.tile(ind_2,(1,len(x),1))==np.tile(np.reshape(np.arange(len(x)),(1,len(x),1)),(len(dx_neigh),1,len(ws)))
        
        dx_all_ext_relevant_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dx_all_ext_relevant_2 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_relevant_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_relevant_2 = -np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        
        dx_all_ext_relevant_1[fil_ind_1] = dx_all_ext_1[fil_ind_1]
        dx_all_ext_relevant_2[fil_ind_2] = dx_all_ext_2[fil_ind_2]
        dy_all_ext_relevant_1[fil_ind_1] = dy_all_ext_1[fil_ind_1]
        dy_all_ext_relevant_2[fil_ind_2] = dy_all_ext_2[fil_ind_2]
        
        dx_ext_relevant_1 = np.amin(dx_all_ext_relevant_1,axis=1)
        dx_ext_relevant_2 = np.amin(dx_all_ext_relevant_2,axis=1)
        dy_ext_relevant_1 = np.amin(dy_all_ext_relevant_1,axis=1)
        dy_ext_relevant_2 = -np.amin(np.abs(dy_all_ext_relevant_2),axis=1)
        
        dx_ext_correction = dx_ext_relevant_1.copy()
        dx_ext_correction[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)] = dx_ext_relevant_2[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)]
        dy_ext_correction = dy_ext_relevant_1.copy()
        dy_ext_correction[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)] = dy_ext_relevant_2[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)]
        
        
        # apply correction considering next waked turbines
        c_t = wind_turbine.ct(np.tile(np.reshape(ws,(1,len(ws))),(len(dx_neigh),1)))
        delta_wd = -((c_t/2)*(np.sin(np.pi*yaw_geom_first_approx/180))*(np.cos(np.pi*yaw_geom_first_approx/180))**2)/((1+k*(dx_ext_correction/diameter))**2)    
            
        fil_correction = (np.abs(dx_ext_correction)!=np.inf) & (np.abs(dy_ext_correction)!=np.inf)
        dx_ext_correction_new = np.inf*np.ones((len(dx_neigh),len(ws)))
        dy_ext_correction_new = np.inf*np.ones((len(dx_neigh),len(ws)))
        dx_ext_correction_new[fil_correction] = dx_ext_correction[fil_correction]*np.cos(delta_wd[fil_correction])+dy_ext_correction[fil_correction]*np.sin(delta_wd[fil_correction])
        dy_ext_correction_new[fil_correction] = -dx_ext_correction[fil_correction]*np.sin(delta_wd[fil_correction])+dy_ext_correction[fil_correction]*np.cos(delta_wd[fil_correction])
        
        yaw_geom_correction = np.sign(dy_ext_correction_new+eps)*yaw_max*((1+p_x)/(p_x+np.e**(dx_ext_correction_new/(q_x*diameter))))*((1+p_y)/(p_y+np.e**(np.abs(dy_ext_correction_new)/(q_y*diameter))))
        yaw_geom_temp = yaw_geom_first_approx+w_corr*yaw_geom_correction
        
        
        # apply correction effective wind speed and set limits
        yaw_geom = np.minimum(np.maximum(f_ws_eff*np.reshape(yaw_geom_temp,(len(dx_neigh),1,len(ws))),-yaw_max),yaw_max)
        
        # assign values
        yaw_temp[fil_wake,:,:] = yaw_geom
        yaw_array[:,wd_ind,:] = np.reshape(yaw_temp,(len(x),len(ws)))
    
    return yaw_array

