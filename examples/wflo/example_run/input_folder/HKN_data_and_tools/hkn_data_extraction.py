#%%
# import packages

import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
import pickle
from numpy import newaxis as na

from py_wake.site import XRSite
import utm


def extract_HKNprice(filename_prices,filename_wind_resources,price_year):
    
    # import databse
    df_prices = pd.read_excel(filename_prices,sheet_name=f'{price_year}')
    df_weather = pd.read_csv(filename_wind_resources)
    
    # extract price data
    price_timeseries = np.array(df_prices['NL'])
    price_timeseries_mean = np.mean(price_timeseries)
    
    # extract weather data (the number indicates the hub height) (first and last days are removed to match price data)
    ws_1_timeseries = np.array(df_weather['WS_1'])[24:8760]
    ws_50_timeseries = np.array(df_weather['WS_50'])[24:8760]
    ws_100_timeseries = np.array(df_weather['WS_100'])[24:8760]
    ws_150_timeseries = np.array(df_weather['WS_150'])[24:8760]
    ws_200_timeseries = np.array(df_weather['WS_200'])[24:8760]
    wd_1_timeseries = np.array(df_weather['WD_1'])[24:8760]
    wd_50_timeseries = np.array(df_weather['WD_50'])[24:8760]
    wd_100_timeseries = np.array(df_weather['WD_100'])[24:8760]
    wd_150_timeseries = np.array(df_weather['WD_150'])[24:8760]
    wd_200_timeseries = np.array(df_weather['WD_200'])[24:8760]
    
    # interpolation at hub height for the wind direction
    hub_height = 170.0 # IEA 22 MW
    wd_timeseries = np.zeros(len(wd_1_timeseries))
    for i in np.arange(len(ws_1_timeseries)):
        height_array = np.array([1,50,100,150,200])
        wd_array = np.array([wd_1_timeseries[i],wd_50_timeseries[i],wd_100_timeseries[i],wd_150_timeseries[i],wd_200_timeseries[i]])
        wd_function = interpolate.interp1d(height_array,wd_array,kind='cubic')
        wd_timeseries[i] = wd_function(hub_height)%360
    
    
    # regression at the hub height (using the shear power law) for the wind speed
    ws_timeseries = np.zeros(len(ws_1_timeseries))
    
    for i in np.arange(len(ws_1_timeseries)):
        
        height_array = np.array([1,50,100,150,200])
        ws_array = np.array([ws_1_timeseries[i],ws_50_timeseries[i],ws_100_timeseries[i],ws_150_timeseries[i],ws_200_timeseries[i]])
        ws_log = np.zeros(len(height_array)-1)
        h_log = np.zeros(len(height_array)-1)
        
        for j in np.arange(len(height_array)-1):
            ws_log[j] = np.log(ws_array[j+1]/ws_array[j])
            h_log[j] = np.log(height_array[j+1]/height_array[j])
    
        # find alpha
        model = LinearRegression()
        model.fit(h_log.reshape(-1, 1),ws_log)
        alpha = model.coef_[0]
        
        # find wind speed at hub height
        ws_timeseries[i] = ws_100_timeseries[i]*(hub_height/100)**alpha
    
    
    # SAMPLE PRICE FOR EACH FLOW CASE (combintation of wd and wd) --------------------------------------------
    # Problem: the results are biased by the fact that there is no price data for the least frequent flow cases
    
    # define wind speed and wind direction bins
    ws_bin_size_price = 5
    wd_bin_size_price = 30
    ws_array_price = np.arange(3,26,ws_bin_size_price)
    wd_array_price = np.arange(0,360,wd_bin_size_price)
    
    # assign to each flow case (combination) the correspondent price
    ws_ind_timeseries = -np.ones(len(ws_timeseries))
    wd_ind_timeseries = -np.ones(len(wd_timeseries))
    ws_lb = ws_array_price-ws_bin_size_price/2
    ws_ub = ws_array_price+ws_bin_size_price/2
    wd_lb = (wd_array_price-wd_bin_size_price/2)%360
    wd_ub = (wd_array_price+wd_bin_size_price/2)%360
    
    for i in np.arange(len(ws_timeseries)):
        if (ws_timeseries[i]>=np.min(ws_array_price))&(ws_timeseries[i]<=np.max(ws_array_price)):   # exclude the case above max ws
            ws_ind_timeseries[i]=np.where(((ws_timeseries[i]>=ws_lb)&(ws_timeseries[i]<ws_ub)))[0][0]
        if (wd_timeseries[i]>=wd_lb[0])|(wd_timeseries[i]<wd_ub[0]):    # check if it is the first wd bin (values around 0deg)
            wd_ind_timeseries[i] = 0
        else:
            wd_ind_timeseries[i]=np.where(((wd_timeseries[i]>=wd_lb)&(wd_timeseries[i]<wd_ub)))[0][0]
    
    # create average price matrix (average price for each flow case)
    price_mat = np.zeros((len(wd_array_price),len(ws_array_price)))
    for i_wd in np.arange(len(wd_array_price)):
        for i_ws in np.arange(len(ws_array_price)):
            fil_wd = wd_ind_timeseries==i_wd
            fil_ws = ws_ind_timeseries==i_ws
            if np.sum(fil_wd&fil_ws)>0:
                price_mat[i_wd,i_ws] = np.mean(price_timeseries[fil_wd&fil_ws])
    
    
    ## SAMPLE PRICE FOR EACH WIND DIRECTION -------------------------------------------------------------
    #
    ## define wind direction bin
    #wd_bin_size_price_wd = 15
    #wd_array_price_wd = np.arange(0,360,wd_bin_size_price_wd)
    #
    ## assign to each wd the correspondent price
    #wd_ind_timeseries = -np.ones(len(wd_timeseries))
    #wd_lb = (wd_array_price_wd-wd_bin_size_price_wd/2)%360
    #wd_ub = (wd_array_price_wd+wd_bin_size_price_wd/2)%360
    #
    #for i in np.arange(len(ws_timeseries)):    
    #    if (wd_timeseries[i]>=wd_lb[0])|(wd_timeseries[i]<wd_ub[0]):    # check if it is the first wd bin (values around 0deg)
    #        wd_ind_timeseries[i] = 0
    #    else:
    #        wd_ind_timeseries[i]=np.where(((wd_timeseries[i]>=wd_lb)&(wd_timeseries[i]<wd_ub)))[0][0]
    #
    ## create average price matrix (average price for each flow case)
    #price_array_wd = np.zeros((len(wd_array_price_wd)))
    #for i_wd in np.arange(len(wd_array_price_wd)):
    #        fil_wd = wd_ind_timeseries==i_wd
    #        if np.sum(fil_wd)>0:
    #            price_array_wd[i_wd] = np.mean(price_timeseries[fil_wd])
    #
    #
    ## SAMPLE PRICE FOR EACH WIND SPEED -------------------------------------------------------------------
    #
    ## define wind direction bin
    #ws_bin_size_price_ws = 1
    #ws_array_price_ws = np.arange(3,26,ws_bin_size_price_ws)
    #
    ## assign to each wd the correspondent price
    #ws_ind_timeseries = -np.ones(len(ws_timeseries))
    #ws_lb = ws_array_price_ws-ws_bin_size_price_ws/2
    #ws_ub = ws_array_price_ws+ws_bin_size_price_ws/2
    #
    #for i in np.arange(len(ws_timeseries)):    
    #    if (ws_timeseries[i]>=np.min(ws_array_price_ws))&(ws_timeseries[i]<=np.max(ws_array_price_ws)):   # exclude the case above max ws
    #        ws_ind_timeseries[i]=np.where(((ws_timeseries[i]>=ws_lb)&(ws_timeseries[i]<ws_ub)))[0][0]
    #
    ## create average price matrix (average price for each flow case)
    #price_array_ws = np.zeros((len(ws_array_price_ws)))
    #for i_ws in np.arange(len(ws_array_price_ws)):
    #        fil_ws = ws_ind_timeseries==i_ws
    #        if np.sum(fil_ws)>0:
    #            price_array_ws[i_ws] = np.mean(price_timeseries[fil_ws])
                
    #return price_mat,wd_array_price,ws_array_price,price_array_wd,wd_array_price_wd,price_array_ws,ws_array_price_ws,price_timeseries_mean
    
    return price_mat,wd_array_price,ws_array_price,price_timeseries_mean



def price_mat_correction(price_mat_lk,
                         price_timeseries_mean,
                         site,
                         wd_array,
                         ws_array
                         ):
    
    weibull_a_array = np.mean(site.ds['Weibull_A'].values,axis=(0,1))
    weibull_k_array = np.mean(site.ds['Weibull_k'].values,axis=(0,1))
    sec_freq_array = np.mean(site.ds['Sector_frequency'].values,axis=(0,1))
    wd_site = site.ds.coords['wd'].values

    p_mat_site = np.zeros((len(wd_site),len(ws_array)))

    for i in np.arange(len(wd_site)):
        p_mat_site[i,:] = sec_freq_array[i]*(np.exp(-((ws_array-0.5)/weibull_a_array[i])**weibull_k_array[i])-np.exp(-((ws_array+0.5)/weibull_a_array[i])**weibull_k_array[i]))

    interp = interpolate.RegularGridInterpolator((wd_site,ws_array),p_mat_site,method='linear',bounds_error=False,fill_value=None)
    WD,WS = np.meshgrid(wd_array,ws_array,indexing='ij')
    p_mat = interp((WD,WS))/(len(wd_array)/len(wd_site))

    price_mat_lk = price_mat_lk*(price_timeseries_mean/np.sum(p_mat*price_mat_lk))

    return price_mat_lk


def generate_HKNscaled_site():

    # extract HKN data
    with open(f'HKN_data.pkl', 'rb') as f:
        HKN_data = pickle.load(f)
    hkn_site = HKN_data['hkn_site']
    diameter = 283.2

    # scale HKN data - boundaries
    coord_sub = utm.from_latlon(52.70,4.29)
    x_sub = coord_sub[0]
    y_sub = coord_sub[1]
    diameter_hkn = 200.

    # scale HKN data - wind resource (create new pywake site object)
    ds_hkn_scaled = xr.Dataset(
        data_vars={
            'Sector_frequency':(['x','y','wd'],hkn_site.ds['Sector_frequency'].values),
            'Weibull_A':(['x','y','wd'],hkn_site.ds['Weibull_A'].values),
            'Weibull_k':(['x','y','wd'],hkn_site.ds['Weibull_k'].values),
            'TI':0.04    
            },
        coords={
            'x':x_sub + (hkn_site.ds['x'].values-x_sub)*(diameter/diameter_hkn),
            'y':y_sub + (hkn_site.ds['y'].values-y_sub)*(diameter/diameter_hkn),
            'wd':hkn_site.ds['wd'].values
            }
        )
    hkn_site_scaled = XRSite(ds_hkn_scaled)

    return hkn_site_scaled


def create_price_mat(price_mat_temp,price_t_mean,wd_array,ws_array,site,wd_array_price,ws_array_price,interp_method='linear'):

    # method to ensure price continuity along the wind direction
    wd_array_price_ext = np.concatenate((wd_array_price,np.array([360.])))
    price_mat_temp_ext = np.concatenate((price_mat_temp,price_mat_temp[0,:][na,:]))

    # interpolation of the price matrix
    interp = interpolate.RegularGridInterpolator((wd_array_price_ext,ws_array_price),price_mat_temp_ext,method=interp_method,bounds_error=False,fill_value=None)
    WD,WS = np.meshgrid(wd_array,ws_array,indexing='ij')
    price_mat_ext = interp((WD,WS))

    # correction to match the average price in the price timeseries (based on the probability of occurence of each bin)
    price_mat_lk = price_mat_correction(price_mat_ext,price_t_mean,site,wd_array,ws_array)
    price_mat_lk_norm = price_mat_lk/price_t_mean

    return price_mat_lk,price_mat_lk_norm



#%%

hkn_site_scaled = generate_HKNscaled_site()
wd_array = np.arange(0,360,2)
ws_array = np.arange(3,26,1)


price_mat_temp_2030,wd_array_price,ws_array_price,price_t_mean_2030 = extract_HKNprice(filename_prices='Price_electricity.xlsx',
                                                                                       filename_wind_resources='weather_ts_hkn.csv',
                                                                                       price_year=2030)

price_mat_2030,price_mat_2030_norm = create_price_mat(price_mat_temp_2030,
                                                      price_t_mean_2030,
                                                      wd_array,
                                                      ws_array,
                                                      hkn_site_scaled,
                                                      wd_array_price,
                                                      ws_array_price,
                                                      interp_method='linear')


price_mat_temp_2040,wd_array_price,ws_array_price,price_t_mean_2040 = extract_HKNprice(filename_prices='Price_electricity.xlsx',
                                                                                       filename_wind_resources='weather_ts_hkn.csv',
                                                                                       price_year=2040)

price_mat_2040,price_mat_2040_norm = create_price_mat(price_mat_temp_2040,
                                                      price_t_mean_2040,
                                                      wd_array,
                                                      ws_array,
                                                      hkn_site_scaled,
                                                      wd_array_price,
                                                      ws_array_price,
                                                      interp_method='linear')



price_mat_temp_2050,wd_array_price,ws_array_price,price_t_mean_2050 = extract_HKNprice(filename_prices='Price_electricity.xlsx',
                                                                                       filename_wind_resources='weather_ts_hkn.csv',
                                                                                       price_year=2050)

price_mat_2050,price_mat_2050_norm = create_price_mat(price_mat_temp_2050,
                                                      price_t_mean_2050,
                                                      wd_array,
                                                      ws_array,
                                                      hkn_site_scaled,
                                                      wd_array_price,
                                                      ws_array_price,
                                                      interp_method='linear')




#%%

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(price_mat_2030,origin='lower')
plt.colorbar()
plt.show()

plt.figure()
plt.plot(wd_array,np.mean(price_mat_2030_norm,axis=(1)),marker='.',label='2030')
plt.plot(wd_array,np.mean(price_mat_2040_norm,axis=(1)),marker='.',label='2040')
plt.plot(wd_array,np.mean(price_mat_2050_norm,axis=(1)),marker='.',label='2050')
plt.legend()
plt.show()

plt.figure()
plt.plot(ws_array,np.mean(price_mat_2030_norm,axis=(0)),marker='.',label='2030')
plt.plot(ws_array,np.mean(price_mat_2040_norm,axis=(0)),marker='.',label='2040')
plt.plot(ws_array,np.mean(price_mat_2050_norm,axis=(0)),marker='.',label='2050')
plt.legend()
plt.show()
#%%

with open(f'HKN_price_data_2deg.pkl', 'wb') as f:
    pickle.dump({'wd_array' : wd_array,
                 'ws_array' : ws_array,
                 'price_mat_2030' : price_mat_2030,
                 'price_mat_2030_norm' : price_mat_2030_norm,
                 'price_mat_2040' : price_mat_2040,
                 'price_mat_2040_norm' : price_mat_2040_norm,
                 'price_mat_2050' : price_mat_2050,
                 'price_mat_2050_norm' : price_mat_2050_norm,
                }, f)




# %%
