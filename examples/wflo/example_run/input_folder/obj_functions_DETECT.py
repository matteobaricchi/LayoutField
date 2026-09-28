# import main packages
import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
from numpy import newaxis as na

# import tools and data
from input_folder.utils.wflop_utils import filter_turbines

# import geometric yaw
from input_folder.geometric_yaw.geometric_yaw import calculate_geomYaw_ExpCorr

# cabling optimization
import networkx as nx
from optiwindnet.api import WindFarmNetwork, HGSRouter, EWRouter

from pathos.multiprocessing import ProcessPool
from functools import partial


class AEPCalculator_DETECT():

    def __init__(self,**kwargs):

        self.wfm = kwargs.get('wfm')
        self.wd_array = kwargs.get('wd_array')
        self.ws_array = kwargs.get('ws_array')
        self.ws_rated = kwargs.get('ws_rated',None)
        self.wind_turbine = kwargs.get('wind_turbine',None)
        self.use_geomYaw = kwargs.get('use_geomYaw',False)
        self.use_inputYaw = kwargs.get('use_inputYaw',False)
        self.yaw_input = kwargs.get('yaw_input',False)
        self.price_lk = kwargs.get('price_lk',None)
        self.price_norm_lk = kwargs.get('price_norm_lk',None)

    def __call__(self,x,y):

        # calculate yaw angles
        if self.use_geomYaw:
            yaw_temp = calculate_geomYaw_ExpCorr(x,y,self.wd_array,self.ws_array,self.ws_rated,self.wind_turbine,self.wfm)
        elif self.use_inputYaw:
            yaw_temp = self.yaw_input
        else:
            yaw_temp = 0

        # compute AEP
        simres = self.wfm(x,y,wd=self.wd_array,ws=self.ws_array,yaw=yaw_temp,tilt=0)
        aep = simres.aep().sum().values
        aep_per_turbine = np.sum(simres.aep().values,axis=(1,2))
        aep_no_wake = simres.aep(with_wake_loss=False).sum().values
        power_ilk = simres.Power.values
        p_ilk = simres.P.values

        # check dimension price data (if does not match set to None)
        if self.price_lk.shape!=(len(self.wd_array),len(self.ws_array)):
            self.price_lk = None
        if self.price_norm_lk.shape!=(len(self.wd_array),len(self.ws_array)):
            self.price_norm_lk = None
        
        # check dimension p_ilk
        if len(p_ilk.shape) < 3:
            if p_ilk.shape == (len(x), len(self.wd_array)):     # only one ws is considered
                p_ilk = p_ilk[:, :, na]
            elif p_ilk.shape == (len(self.wd_array), len(self.ws_array)):   # average wind resource
                p_ilk = p_ilk[na, :, :]
            else:
                raise TypeError('Case not considered - fix dimension p_ilk')

        # compute valued AEP
        if self.price_norm_lk is None:
            self.price_norm_lk = np.zeros((len(self.wd_array),len(self.ws_array)))
        vaep_per_turbine = np.sum(self.price_norm_lk[na,:,:]*p_ilk*power_ilk*8760/1e9,axis=(1,2))
        vaep = np.sum(vaep_per_turbine)

        # compute REVENUES
        if self.price_lk is None:
            self.price_lk = np.zeros((len(self.wd_array),len(self.ws_array)))
        rev_per_turbine = np.sum(self.price_lk[na,:,:]*p_ilk*power_ilk*8760/1e9,axis=(1,2))
        rev = np.sum(rev_per_turbine)

        # save data
        ds = xr.Dataset(
            {
                "x": ('i', x),
                "y": ('i', y),
                "wd_array": ('l', self.wd_array),
                "ws_array": ('k', self.ws_array),
                "power_ilk": (('i','l','k'), power_ilk),
                "AEP_per_turbine": ('i', aep_per_turbine),
                "vAEP_per_turbine": ('i', vaep_per_turbine),
                "REV_per_turbine": ('i', rev_per_turbine),
                "AEP": aep,
                "vAEP": vaep,
                "REV": rev,
                "AEP_no_wake": aep_no_wake,
            },
            coords={
                "i": np.arange(len(x)),
                "l": np.arange(len(self.wd_array)),
                "k": np.arange(len(self.ws_array)),
            }
        )

        return ds
    



class AEPCalculator_DETECT_Weibull():

    # some functionalities have not been implemented (e.g. revenues, valued-aep, yaw)

    def __init__(self,**kwargs):

        self.wfm = kwargs.get('wfm')
        self.wd_array = kwargs.get('wd_array')
        self.ws_array = kwargs.get('ws_array')
        self.ws_rated = kwargs.get('ws_rated',None)
        self.wind_turbine = kwargs.get('wind_turbine',None)
        self.use_geomYaw = kwargs.get('use_geomYaw',False)
        self.use_inputYaw = kwargs.get('use_inputYaw',False)
        self.yaw_input = kwargs.get('yaw_input',False)
        self.price_lk = kwargs.get('price_lk',None)
        self.price_norm_lk = kwargs.get('price_norm_lk',None)
        self.x_sub = kwargs.get('x_sub',0.)
        self.y_sub = kwargs.get('y_sub',0.)
        self.samps = kwargs.get('samps',100)

        # extarct flow statistics (based on the position of the substation)
        self.freqs = self.wfm.site.local_wind(self.x_sub,self.y_sub,wd=self.wd_array,ws=self.ws_array).Sector_frequency_ilk[0,:,0]
        self.As = self.wfm.site.local_wind(self.x_sub,self.y_sub,wd=self.wd_array,ws=self.ws_array).Weibull_A_ilk[0,:,0]
        self.ks = self.wfm.site.local_wind(self.x_sub,self.y_sub,wd=self.wd_array,ws=self.ws_array).Weibull_k_ilk[0,:,0]


    def __call__(self,x,y):

        # flow sampling
        idx = np.random.choice(np.arange(self.wd_array.size),self.samps,p=self.freqs/np.sum(self.freqs))
        wd_t = self.wd_array[idx]
        A = self.As[idx]
        k = self.ks[idx]
        ws_t = A * np.random.weibull(k)

        # compute AEP
        simres = self.wfm(x,y,wd=wd_t,ws=ws_t,yaw=0,tilt=0,time=True)
        aep = simres.aep().sum().values
        aep_per_turbine = np.sum(simres.aep().values,axis=(1))
        aep_no_wake = simres.aep(with_wake_loss=False).sum().values

        # save data
        ds = xr.Dataset(
            {
                "x": ('i', x),
                "y": ('i', y),
                "wd_array": ('l', self.wd_array),
                "ws_array": ('k', self.ws_array),
                "power_ilk": (('i','l','k'), np.zeros((len(x),len(self.wd_array),len(self.ws_array)))),  # implementation not availbale yet
                "AEP_per_turbine": ('i', aep_per_turbine),
                "vAEP_per_turbine": ('i', np.zeros_like(aep_per_turbine)), # implementation not availbale yet
                "REV_per_turbine": ('i', np.zeros_like(aep_per_turbine)),  # implementation not availbale yet
                "AEP": aep,
                "vAEP": 0., # implementation not availbale yet
                "REV": 0.,  # implementation not availbale yet
                "AEP_no_wake": aep_no_wake,
            },
            coords={
                "i": np.arange(len(x)),
                "l": np.arange(len(self.wd_array)),
                "k": np.arange(len(self.ws_array)),
            }
        )

        return ds




class CableCalculator_DETECT():

    def __init__(self,**kwargs):
        self.time_limit = kwargs.get('time_limit',0.3)
        self.x_sub = kwargs.get('x_sub')
        self.y_sub = kwargs.get('y_sub')
        cable_specs = kwargs.get('cable_specs')
        self.cables = np.array([(int(c["capacity_NrT"]), float(c["cost_€_m"])) for c in cable_specs],dtype=[("capacity", int), ("cost", float)])
        #self.x_init = kwargs.get('x_init')
        #self.y_init = kwargs.get('y_init')


    def Postprocess_Cable_Optimizer(self,wfn):
        cab_data = wfn.get_network()
        # get connection matrix
        u_fnt = []
        v_fnt = []
        fnT = wfn.G.graph.get('fnT', None)
        fnT_state = fnT is None
        for u, v in wfn.G.edges():
             u_fnt.append(u if fnT_state else fnT[u])
             v_fnt.append(v if fnT_state else fnT[v])
        
        # Postprocess to deal with obstacles
        # Step 1: Build undirected graph
        H = nx.Graph()
        length_lookup = {}
        type_lookup = {}
        load_lookup = {}
        idx_ct = 4
        substation = -1
        for row in cab_data:
            a, b = int(row[0]), int(row[1])
            length = row[2]
            ct = row[idx_ct]
            l = row[3]
            H.add_edge(a, b)
            length_lookup[(a, b)] = length
            length_lookup[(b, a)] = length  # undirected
            type_lookup[(a, b)] = ct
            type_lookup[(b, a)] = ct  # undirected
            load_lookup[(a, b)] = l
            load_lookup[(b, a)] = l  # undirected
        # Step 2: Traverse from substation
        visited = set()
        edges = []
        lengths = []
        types = []
        loads = []
        def dfs(node, parent=None):
            visited.add(node)
            for neighbor in H.neighbors(node):
                if neighbor not in visited:
                    # Reverse direction to point to substation
                    edges.append((neighbor, node))
                    lengths.append(length_lookup[(neighbor, node)])
                    types.append(type_lookup[(neighbor, node)])
                    loads.append(load_lookup[(neighbor, node)])
                    dfs(neighbor, node)
        
        dfs(substation)
        
        # Now `edges` contains tuples of form (source, target) pointing to substation
        source_nodes = [int(src) for src, dst in edges]
        target_nodes = [int(dst) for src, dst in edges]
        lengths = [float(l) for l in lengths]
        types = [int(t) for t in types]
        loads = [int(t) for t in loads]
        
        # Step 3: Combine and sort by source
        combined = sorted(zip(source_nodes, target_nodes, loads), key=lambda x: x[0])
        s, t, load_sorted = zip(*combined)
        return t, [int(x) for x in u_fnt], [int(x) for x in v_fnt], np.array([cab_data['src'].tolist(), cab_data['tgt'].tolist()])


    def __call__(self,x,y):
        # First try metaheuristic solver, if it fails, go with heuristics.
        try:
            wfn = WindFarmNetwork(turbinesC=np.column_stack((x,y)), substationsC=np.column_stack((self.x_sub, self.y_sub)), cables=self.cables, router=HGSRouter(time_limit=self.time_limit))
            wfn.optimize()
        except:
            #x_init_fil = self.x_init[:len(x)]
            #y_init_fil = self.y_init[:len(y)]
            print("Metaheuristic solver failed, trying with heuristics...")
            wfn = WindFarmNetwork(turbinesC=np.column_stack((x,y)), substationsC=np.column_stack((self.x_sub, self.y_sub)), cables=self.cables, router=EWRouter())
            wfn.optimize()
        TurCon, u, v, cab_data = self.Postprocess_Cable_Optimizer(wfn)

        ds = xr.Dataset(
            {
                "x": ('i', x),
                "y": ('i', y),
                "TurCon": ('c', np.array(TurCon)),
                "u": ('c', u),
                "v": ('c', v),
                "cab_data": (('c_type','c'), cab_data),
            },
            coords={
                "i": np.arange(len(x)),
                "c": np.arange(len(TurCon)),
                "c_type": np.arange(cab_data.shape[0]),
            }
        )

        return ds



class WaterDepthCalculator_DETECT():

    def __init__(self,**kwargs):
        self.site_x_grid = kwargs.get('site_x_grid')
        self.site_y_grid = kwargs.get('site_y_grid')
        self.site_bathymetry_grid = kwargs.get('site_bathymetry_grid')

    def __call__(self,x,y):
        site_x_coord = self.site_x_grid[:,0]
        site_y_coord = self.site_y_grid[0,:]
        interp_function = RegularGridInterpolator((site_x_coord,site_y_coord),self.site_bathymetry_grid)
        water_depth = interp_function((x,y))    
        return water_depth











# python calculators (define as global)
AEP_CALC = None
CABLE_CALC = None
WATER_DEPTH_CALC = None
MIN_D = None

def init_worker(aep_calculator,cable_calculator,water_depth_calculator,min_d):
    global AEP_CALC, CABLE_CALC, WATER_DEPTH_CALC, MIN_D
    AEP_CALC = aep_calculator
    CABLE_CALC = cable_calculator
    WATER_DEPTH_CALC = water_depth_calculator
    MIN_D = min_d


# python function wrapper
def python_func_wrapper(args):
    x, y = args
    x_fil,y_fil,ind_keep = filter_turbines(x,y,MIN_D)
    ds_aep = AEP_CALC(x_fil,y_fil)
    if CABLE_CALC is None:
        ds_cable = None
    else:
        ds_cable = CABLE_CALC(x_fil, y_fil)
    water_depth_per_turbine = WATER_DEPTH_CALC(x_fil, y_fil)
    return {
        'x_fil': x_fil,
        'y_fil': y_fil,
        'ind_keep': ind_keep,
        'ds_aep': ds_aep,
        'ds_cable': ds_cable,
        'water_depth_per_turbine': water_depth_per_turbine,
    }

class PopEval_python_XYwrapper():

    def __init__(self,aep_calculator,cable_calculator,water_depth_calculator,min_d,**kwargs):

        self.aep_calculator = aep_calculator
        self.cable_calculator = cable_calculator
        self.water_depth_calculator = water_depth_calculator
        self.min_d = min_d

        # attributes for optmization
        self.parallel_execution = kwargs.get('parallel_execution',False)
        self.n_cpu = kwargs.get('n_cpu',None)


    def __call__(self,x_mat,y_mat):
        
        # extract dimension
        if len(x_mat.shape)<2:
            raise TypeError('Population not in the right input format: check dimensions -> x: (n_pop,n_wt)')
        n_pop = x_mat.shape[0]
        n_wt = x_mat.shape[1]

        # run python wrapper
        if self.parallel_execution:
            pool = ProcessPool(self.n_cpu,
                               initializer=init_worker,
                               initargs=(self.aep_calculator,
                                         self.cable_calculator,
                                         self.water_depth_calculator,
                                         self.min_d))
            results =  pool.map(python_func_wrapper,zip(list(x_mat),list(y_mat)))
        else:
            init_worker(
                self.aep_calculator,
                self.cable_calculator,
                self.water_depth_calculator,
                self.min_d
            )
            results = map(python_func_wrapper,zip(list(x_mat),list(y_mat)))

        # unpack results
        res_list = list(results)
        x_fil_list = [None]*n_pop
        y_fil_list = [None]*n_pop
        ind_keep_list = [None]*n_pop
        ds_aep_list = [None]*n_pop
        ds_cable_list = [None]*n_pop
        water_depth_per_turbine_list = [None]*n_pop
        for n in np.arange(n_pop):
            x_fil_list[n] = res_list[n]['x_fil']
            y_fil_list[n] = res_list[n]['y_fil']
            ind_keep_list[n] = res_list[n]['ind_keep']
            ds_aep_list[n] = res_list[n]['ds_aep']
            ds_cable_list[n] = res_list[n]['ds_cable']
            water_depth_per_turbine_list[n] = res_list[n]['water_depth_per_turbine']

        # save output
        dict_python_output = {
            'x_fil_list' : x_fil_list,
            'y_fil_list' : y_fil_list,
            'ind_keep_list' : ind_keep_list,
            'ds_aep_list' : ds_aep_list,
            'ds_cable_list' : ds_cable_list,
            'water_depth_per_turbine_list' : water_depth_per_turbine_list,
        }

        return dict_python_output













class PopEval_matlab_XYwrapper():

    def __init__(self,matlab_eng,**kwargs):

        self.matlab_eng = matlab_eng

        # attributes for optmization
        self.parallel_execution = kwargs.get('parallel_execution',False)
        self.n_cpu = kwargs.get('n_cpu',None)


    def __call__(self,dict_python_output):

        # unpack python output
        x_fil_list = dict_python_output['x_fil_list']
        y_fil_list = dict_python_output['y_fil_list']
        ind_keep_list = dict_python_output['ind_keep_list']

        n_pop = len(x_fil_list)
        z_fil_list = [None]*n_pop
        aep_input_list = [None]*n_pop
        aep_no_wake_list = [None]*n_pop
        power_ilk_list = [None]*n_pop
        TurCon_list = [None]*n_pop
        u_list = [None]*n_pop
        v_list = [None]*n_pop
        cab_data_list = [None]*n_pop
        wd_step_list = [None]*n_pop
        for n in np.arange(n_pop):
            z_fil_list[n] = -dict_python_output['water_depth_per_turbine_list'][n]
            aep_input_list[n] = dict_python_output['ds_aep_list'][n]['AEP'].values
            aep_no_wake_list[n] = dict_python_output['ds_aep_list'][n]['AEP_no_wake'].values
            power_ilk_list[n] = dict_python_output['ds_aep_list'][n]['power_ilk'].values
            wd_step_list[n] = float(np.diff(dict_python_output['ds_aep_list'][n]['wd_array'].values)[0])
            TurCon_list[n] = dict_python_output['ds_cable_list'][n]['TurCon'].values
            u_list[n] = dict_python_output['ds_cable_list'][n]['u'].values
            v_list[n] = dict_python_output['ds_cable_list'][n]['v'].values
            cab_data_list[n] = dict_python_output['ds_cable_list'][n]['cab_data'].values

        # process these data with the matlab engine running DETECT for the all the elements in the lists
        lcoe_list,ioe_list,cost_per_turbine_list,emission_per_turbine_list,aep_net_factor_list = self.matlab_eng.evaluate_population(x_fil_list,
                                                                                                                                     y_fil_list,
                                                                                                                                     z_fil_list,
                                                                                                                                     aep_input_list,
                                                                                                                                     aep_no_wake_list,
                                                                                                                                     power_ilk_list,
                                                                                                                                     wd_step_list,
                                                                                                                                     TurCon_list,
                                                                                                                                     u_list,
                                                                                                                                     v_list,
                                                                                                                                     cab_data_list,
                                                                                                                                     self.parallel_execution,
                                                                                                                                     self.n_cpu,
                                                                                                                                     nargout=5)
        # postprocess matlab variables
        for n in np.arange(n_pop):
            cost_per_turbine_list[n] = np.array(cost_per_turbine_list[n]).reshape(-1)
            emission_per_turbine_list[n] = np.array(emission_per_turbine_list[n]).reshape(-1)
        
        # save output
        dict_matlab_output = {
            'lcoe_list' : lcoe_list,
            'ioe_list' : ioe_list,
            'cost_per_turbine_list' : cost_per_turbine_list,
            'emission_per_turbine_list' : emission_per_turbine_list,
            'aep_net_factor_list' : aep_net_factor_list,
        }

        return dict_matlab_output


def calculate_soft_penalty_factor(x,y,gen_number_norm,min_d_bounds,method_gen_weight='linear',method_min_d_weight='linear'):
    # values per turbines:
    # 1: max violation of the min d constraint
    # 0: no violation of the min d constraint

    x_mat_1 = np.tile(np.reshape(x,(len(x),1)),(1,len(x)))
    x_mat_2 = np.tile(np.reshape(x,(1,len(x))),(len(x),1))
    y_mat_1 = np.tile(np.reshape(y,(len(y),1)),(1,len(y)))
    y_mat_2 = np.tile(np.reshape(y,(1,len(y))),(len(y),1))
    d = np.sqrt((x_mat_1-x_mat_2)**2+(y_mat_1-y_mat_2)**2)
    np.fill_diagonal(d, np.inf)
    min_d_array = np.min(d,axis=1)

    if method_gen_weight=='linear':
        coeff_gen_number = gen_number_norm
    elif method_gen_weight=='poly2':
        coeff_gen_number = gen_number_norm**2
    elif method_gen_weight=='poly5':
        coeff_gen_number = gen_number_norm**5
    else:
        TypeError('Penalty method (gen weigth) not supported')

    if method_min_d_weight=='bool':
        coeff_min_d_array = np.zeros(len(x))
        coeff_min_d_array[min_d_array<min_d_bounds[1]] = 1.
    if method_min_d_weight=='linear':
        coeff_min_d_array = np.clip((min_d_bounds[1]-min_d_array)/(min_d_bounds[1]-min_d_bounds[0]),0,1)
    else:
        TypeError('Penalty method (min d weigth) not supported')

    return coeff_min_d_array*coeff_gen_number

    



class ObjFunction_XYwrapper():

    def __init__(self,python_xy_wrapper,matlab_xy_wrapper,**kwargs):
        
        self.python_xy_wrapper = python_xy_wrapper
        self.matlab_xy_wrapper = matlab_xy_wrapper

        self.value_per_turbine = kwargs.get('value_per_turbine',False)
        self.output_keys = kwargs.get('output_keys',['AEP','LCOE','IOE'])
        self.n_obj = len(self.output_keys)
        self.maximize = kwargs.get('maximize',[True]*self.n_obj)

        self.min_d_bounds = kwargs.get('min_d_bounds',[1,0])


    def __call__(self,x_mat,y_mat,gen_number_norm=1.):

        # initialize results
        n_pop,n_wt = x_mat.shape
        aep_array = np.zeros((n_pop))
        rev_array = np.zeros((n_pop))
        aep_net_array = np.zeros((n_pop))
        vaep_net_array = np.zeros((n_pop))
        lcoe_array = np.zeros((n_pop))
        ioe_array = np.zeros((n_pop))
        aep_per_turbine_mat = np.ones((n_pop,n_wt))*1e-10
        rev_per_turbine_mat = np.ones((n_pop,n_wt))*1e-10
        aep_net_per_turbine_mat = np.ones((n_pop,n_wt))*1e-10
        vaep_net_per_turbine_mat = np.ones((n_pop,n_wt))*1e-10
        cost_per_turbine_mat = np.ones((n_pop,n_wt))*1e20
        emission_per_turbine_mat = np.ones((n_pop,n_wt))*1e20
        eps = 1e-6

        # run python and matlab functions
        dict_python_output = self.python_xy_wrapper(x_mat,y_mat)
        if self.matlab_xy_wrapper is None:
            dict_matlab_output = None
        else:
            dict_matlab_output = self.matlab_xy_wrapper(dict_python_output)

        # postprocess results
        for n in np.arange(n_pop):
            ind_keep = dict_python_output['ind_keep_list'][n]
            aep_array[n] = dict_python_output['ds_aep_list'][n]['AEP'].values
            rev_array[n] = dict_python_output['ds_aep_list'][n]['REV'].values
            aep_per_turbine_mat[n,ind_keep] = dict_python_output['ds_aep_list'][n]['AEP_per_turbine'].values
            rev_per_turbine_mat[n,ind_keep] = dict_python_output['ds_aep_list'][n]['REV_per_turbine'].values
            if self.matlab_xy_wrapper is not None:
                aep_net_array[n] = dict_python_output['ds_aep_list'][n]['AEP'].values*dict_matlab_output['aep_net_factor_list'][n]
                vaep_net_array[n] = dict_python_output['ds_aep_list'][n]['vAEP'].values*dict_matlab_output['aep_net_factor_list'][n]
                lcoe_array[n] = dict_matlab_output['lcoe_list'][n]
                ioe_array[n] = dict_matlab_output['ioe_list'][n]
                aep_net_per_turbine_mat[n,ind_keep] = dict_python_output['ds_aep_list'][n]['AEP_per_turbine'].values*dict_matlab_output['aep_net_factor_list'][n]
                vaep_net_per_turbine_mat[n,ind_keep] = dict_python_output['ds_aep_list'][n]['vAEP_per_turbine'].values*dict_matlab_output['aep_net_factor_list'][n]
                cost_per_turbine_mat[n,ind_keep] = dict_matlab_output['cost_per_turbine_list'][n]
                emission_per_turbine_mat[n,ind_keep] = dict_matlab_output['emission_per_turbine_list'][n]

        lcoe_per_turbine_mat = (np.sum(cost_per_turbine_mat/1000,axis=(1))/np.sum(aep_net_per_turbine_mat,axis=(1))/n_wt)[:,na] + eps*(cost_per_turbine_mat/1000/aep_net_per_turbine_mat)
        ioe_per_turbine_mat = (np.sum(emission_per_turbine_mat/1000,axis=(1))/np.sum(aep_net_per_turbine_mat,axis=(1))/n_wt)[:,na] + eps*(emission_per_turbine_mat/1000/aep_net_per_turbine_mat)
        cove_per_turbine_mat = (np.sum(cost_per_turbine_mat/1000,axis=(1))/np.sum(vaep_net_per_turbine_mat,axis=(1))/n_wt)[:,na] + eps*(cost_per_turbine_mat/1000/vaep_net_per_turbine_mat)
        cove_array = np.sum(cost_per_turbine_mat/1000,axis=(1))/np.sum(vaep_net_per_turbine_mat,axis=(1))


        # apply soft penalty on minimum distance
        if self.min_d_bounds[1]>self.min_d_bounds[0]:

            for n in np.arange(n_pop):
                soft_penalty_per_turbine = calculate_soft_penalty_factor(x_mat[n,:],y_mat[n,:],gen_number_norm,self.min_d_bounds,method_min_d_weight='bool',method_gen_weight='poly2')
                aep_per_turbine_mat[n,:] = aep_per_turbine_mat[n,:]*(1-soft_penalty_per_turbine)+1e-10
                rev_per_turbine_mat[n,:] = rev_per_turbine_mat[n,:]*(1-soft_penalty_per_turbine)+1e-10
                aep_net_per_turbine_mat[n,:] = aep_net_per_turbine_mat[n,:]*(1-soft_penalty_per_turbine)+1e-10
                vaep_net_per_turbine_mat[n,:] = vaep_net_per_turbine_mat[n,:]*(1-soft_penalty_per_turbine)+1e-10

            # for these metrics the penalty is applied only on the denominator
            lcoe_per_turbine_mat = (np.sum(cost_per_turbine_mat/1000,axis=(1))/np.sum(aep_net_per_turbine_mat,axis=(1))/n_wt)[:,na] + eps*(cost_per_turbine_mat/1000/aep_net_per_turbine_mat)
            ioe_per_turbine_mat = (np.sum(emission_per_turbine_mat/1000,axis=(1))/np.sum(aep_net_per_turbine_mat,axis=(1))/n_wt)[:,na] + eps*(emission_per_turbine_mat/1000/aep_net_per_turbine_mat)
            cove_per_turbine_mat = (np.sum(cost_per_turbine_mat/1000,axis=(1))/np.sum(vaep_net_per_turbine_mat,axis=(1))/n_wt)[:,na] + eps*(cost_per_turbine_mat/1000/vaep_net_per_turbine_mat)
            
            aep_array = np.sum(aep_per_turbine_mat,axis=1)
            rev_array = np.sum(rev_per_turbine_mat,axis=1)
            aep_net_array = np.sum(aep_net_per_turbine_mat,axis=1)
            vaep_net_array = np.sum(vaep_net_per_turbine_mat,axis=1)
            lcoe_array = np.sum(cost_per_turbine_mat/1000,axis=(1))/np.sum(aep_net_per_turbine_mat,axis=(1))
            ioe_array = np.sum(emission_per_turbine_mat/1000,axis=(1))/np.sum(aep_net_per_turbine_mat,axis=(1))
            cove_array = np.sum(cost_per_turbine_mat/1000,axis=(1))/np.sum(vaep_net_per_turbine_mat,axis=(1))


        # save output
        dict_output = {
            'AEP' : aep_array,
            'REV' : rev_array,
            'AEPnet' : aep_net_array,
            'vAEPnet' : vaep_net_array,
            'LCOE' : lcoe_array,
            'IOE' : ioe_array,
            'COVE' : cove_array,
            'AEP_per_turbine' : aep_per_turbine_mat,
            'REV_per_turbine' : rev_per_turbine_mat,
            'AEPnet_per_turbine' : aep_net_per_turbine_mat,
            'vAEPnet_per_turbine' : vaep_net_per_turbine_mat,
            'LCOE_per_turbine' : lcoe_per_turbine_mat,
            'IOE_per_turbine' : ioe_per_turbine_mat,
            'COVE_per_turbine' : cove_per_turbine_mat,
        }

        # define sign
        sign_array = np.where(np.array(self.maximize),1,-1)

        # select output
        if self.value_per_turbine:
            f_out = np.zeros((n_pop,n_wt,self.n_obj))
            for n_obj_ind in np.arange(self.n_obj):
                f_out[:,:,n_obj_ind] = dict_output[self.output_keys[n_obj_ind]+'_per_turbine']
            f_out = f_out*sign_array[na,na,:]
        else:
            f_out = np.zeros((n_pop,self.n_obj))
            for n_obj_ind in np.arange(self.n_obj):
                f_out[:,n_obj_ind] = dict_output[self.output_keys[n_obj_ind]]
            f_out = f_out*sign_array[na,:]

        # fix dimensions
        if self.value_per_turbine:
            if self.n_obj<2:
                f_out = np.reshape(f_out,(n_pop,n_wt))
        else:
            if self.n_obj<2:
                f_out = np.reshape(f_out,(n_pop))
            else:
                f_out = np.reshape(f_out,(n_pop,self.n_obj))

        return f_out







