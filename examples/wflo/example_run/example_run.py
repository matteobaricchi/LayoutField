#%%

import numpy as np
import time
import pickle
from numpy import newaxis as na
import utm

# import LO-GA
from input_folder.loga.loga_par_matlab_v2 import Boundaries
from input_folder.loga.loga_par_matlab_v2 import LayoutOptimizationGA

# import utils
from input_folder.utils.wflop_utils import randomize_layouts
#from optiwindnet.augmentation import poisson_disc_filler

# import obj function wrappers
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPTurbOPark
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPBastankhah
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPZong
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPJensen
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPTurbOParkLsum
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPTurbOParkRcenter
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPTurbOParkCGI
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPTurbOPark1deg
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPTurbOPark5deg
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPTurbOPark10deg
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPTurbOPark8ms
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPTurbOParkaWR
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_REVTurbOPark
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_LCOETurbOPark
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_IOETurbOPark
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_COVETurbOPark
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPTurbOParkGY
from input_folder.obj_funcs_v2 import ObjFunc_HKNscaled_AEPTurbOParkCab

from input_folder.utils.wflop_utils import min_distance

if __name__ == '__main__':

    # extract HKN data
    with open(f'input_folder/HKN_data_and_tools/HKN_data.pkl', 'rb') as f:
        HKN_data = pickle.load(f)
    hkn_boundaries_x = HKN_data['hkn_boundaries_x']
    hkn_boundaries_y = HKN_data['hkn_boundaries_y']
    hkn_wt_x = HKN_data['hkn_wt_x']
    hkn_wt_y = HKN_data['hkn_wt_y']
    diameter = 283.2
    n_wt = len(hkn_wt_x)

    # define scaling factor (used for case studies with different power densities)
    f_scaling = 1.

    # scale HKN data - boundaries and initial positions
    coord_sub = utm.from_latlon(52.70,4.29)
    x_sub = coord_sub[0]
    y_sub = coord_sub[1]
    diameter_hkn = 200.
    hkn_wt_x_scaled = x_sub + (hkn_wt_x-x_sub)*(diameter/diameter_hkn)*f_scaling
    hkn_wt_y_scaled = y_sub + (hkn_wt_y-y_sub)*(diameter/diameter_hkn)*f_scaling
    hkn_boundaries_x_scaled = x_sub + (hkn_boundaries_x-x_sub)*(diameter/diameter_hkn)*f_scaling
    hkn_boundaries_y_scaled = y_sub + (hkn_boundaries_y-y_sub)*(diameter/diameter_hkn)*f_scaling
    boundaries = Boundaries([hkn_boundaries_x_scaled],[hkn_boundaries_y_scaled])

    # optimization parameters
    n_pop = 350                         # size of the population
    n_gen = 800                        # number of generations
    n_keep_parents = int(0.3*n_pop)     # number of parents kept from the previous generation
    p_s_min = 0.7                       # min percentage of selection
    p_s_max = 0.7                       # max percentage of selection
    p_m_min = 0.1                       # min percentage of mutation
    p_m_max = 0.3                       # max percentage of mutation
    s_m_min = 0*diameter                # min step of mutation
    s_m_max = 3*diameter                # max step of mutation
    d_limit = 1*diameter                # distance limit for turbine association during crossover
    p_m_array = np.flip(np.linspace(p_m_min,p_m_max,n_gen))
    s_m_array = s_m_min+(s_m_max-s_m_min)*(np.exp(-5*np.linspace(0,1,n_gen)))

    # extract initial population
    with open(f'input_folder/initial_layouts/initial_pop_v1.pkl', 'rb') as f:
        data_init_pop = pickle.load(f)
    x_mat_initial = x_sub + (data_init_pop['x_mat_initial'][:n_pop,:]-x_sub)*f_scaling
    y_mat_initial = y_sub + (data_init_pop['y_mat_initial'][:n_pop,:]-y_sub)*f_scaling

    # define objective function wrapper
    obj_func = ObjFunc_HKNscaled_AEPTurbOPark5deg(n_cpu=48,
                                            parallel_execution=True
                                            )

    # create optimization object (SINGLE-OBJECTIVE)
    layout_optimization = LayoutOptimizationGA(obj_func,  
                                               n_wt,
                                               boundaries,
                                               n_gen,
                                               n_pop,
                                               perc_s=0.7,
                                               perc_m=p_m_array,
                                               step_m=s_m_array,
                                               distance_limit=d_limit,
                                               full_pop_evaluation=True,
                                               x_mat_initial=x_mat_initial,
                                               y_mat_initial=y_mat_initial,
                                               gen_to_f=True
                                               )
    
    # run optmization
    print('Single-objective optimization (parallel)')
    t_1 = time.time()
    x_opt,y_opt,f_opt,_,f_opt_gen = layout_optimization.optimize()
    t_2 = time.time()
    print(f'Total time: {t_2-t_1}')

    # calculate min distance to check convergence
    min_d_D_opt = min_distance(x_opt,y_opt,diameter)


    # postprocess optimization results

    data_results = {}
    data_results['x_opt'] = x_opt
    data_results['y_opt'] = y_opt
    data_results['x_mat_initial']  = hkn_wt_x_scaled
    data_results['y_mat_initial'] = hkn_wt_y_scaled
    data_results['f_opt'] = f_opt
    data_results['f_opt_gen'] = f_opt_gen
    data_results['converged_flag'] = min_d_D_opt<=3.5
    data_results['min_d_D_opt'] = min_d_D_opt

    f_wrapper_list = [ObjFunc_HKNscaled_AEPTurbOPark(),
                      ObjFunc_HKNscaled_AEPBastankhah(),
                      ObjFunc_HKNscaled_AEPZong(),
                      ObjFunc_HKNscaled_AEPJensen(),
                      ObjFunc_HKNscaled_AEPTurbOParkLsum(),
                      ObjFunc_HKNscaled_AEPTurbOParkRcenter(),
                      ObjFunc_HKNscaled_AEPTurbOParkCGI(),
                      ObjFunc_HKNscaled_AEPTurbOPark1deg(),
                      ObjFunc_HKNscaled_AEPTurbOPark5deg(),
                      ObjFunc_HKNscaled_AEPTurbOPark10deg(),
                      ObjFunc_HKNscaled_AEPTurbOPark8ms(),
                      ObjFunc_HKNscaled_AEPTurbOParkaWR(),
                      ObjFunc_HKNscaled_LCOETurbOPark(),  # not actual value (contrubution of each turbine is summed mulitplied by eps)
                      ObjFunc_HKNscaled_IOETurbOPark(),   # not actual value (contrubution of each turbine is summed mulitplied by eps)
                      ObjFunc_HKNscaled_REVTurbOPark(),
                      ObjFunc_HKNscaled_COVETurbOPark(),  # not actual value (contrubution of each turbine is summed mulitplied by eps)
                      ObjFunc_HKNscaled_AEPTurbOParkGY(),
                      ObjFunc_HKNscaled_AEPTurbOParkCab()
    ]

    f_name_list = ['AEPTurbOPark',
                   'AEPBastankhah',
                   'AEPZong',
                   'AEPJensen',
                   'AEPTurbOParkLsum',
                   'AEPTurbOParkRcenter',
                   'AEPTurbOParkCGI',
                   'AEPTurbOPark1deg',
                   'AEPTurbOPark5deg',
                   'AEPTurbOPark10deg',
                   'AEPTurbOPark8ms',
                   'AEPTurbOParkaWR',
                   'LCOETurbOPark',
                   'IOETurbOPark',
                   'REVTurbOPark',
                   'COVETurbOPark',
                   'AEPTurbOParkGY',
                   'AEPTurbOParkCab'
    ]

    print('Postprocessing...')
    t_1 = time.time()
    for name,wrapper in zip(f_name_list,f_wrapper_list):
        f_eval = wrapper
        data_results[name] = np.sum(f_eval(x_opt[na,:],y_opt[na,:]))
    t_2 = time.time()
    print(f'Total time: {t_2-t_1}')

    # store data
    with open('data_HKNscaled_AEPTurbOPark5deg_v1.pkl', 'wb') as f:
        pickle.dump(data_results, f)





















