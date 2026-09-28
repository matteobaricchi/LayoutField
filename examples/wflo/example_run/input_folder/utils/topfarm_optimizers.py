#%%

import numpy as np
import time

# import py_wake packages needed for SGD
from py_wake.utils.gradients import autograd
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasySGDDriver,EasyScipyOptimizeDriver,EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from topfarm.plotting import NoPlot
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
#from topfarm.recorders import TopFarmListRecorder
#from topfarm.constraint_components.constraint_aggregation import ConstraintAggregation
from topfarm.constraint_components.constraint_aggregation import DistanceConstraintAggregation



class LayoutOptimizationSGD():

    def __init__(self,
                 n_wt,
                 wfm,
                 diameter,
                 wd_array,
                 ws_array,
                 x_sub,
                 y_sub,
                 boundary,
                 maxiter,
                 learning_rate,
                 samps=100):
        
        self.n_wt = n_wt
        self.wfm = wfm
        self.diameter = diameter
        self.wd_array = wd_array
        self.ws_array = ws_array
        self.x_sub = x_sub
        self.y_sub = y_sub
        self.boundary = boundary
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.samps = samps

        # initialize wind resource and farm parameters
        freqs = self.wfm.site.local_wind(np.array([x_sub]),np.array([y_sub]),wd=self.wd_array,ws=self.ws_array).Sector_frequency_ilk[0,:,0]     #sector frequency
        self.As = self.wfm.site.local_wind(np.array([x_sub]),np.array([y_sub]),wd=self.wd_array,ws=self.ws_array).Weibull_A_ilk[0,:,0]               #weibull A
        self.ks = self.wfm.site.local_wind(np.array([x_sub]),np.array([y_sub]),wd=self.wd_array,ws=self.ws_array).Weibull_k_ilk[0,:,0]              #weibull k
        self.freqs_norm = freqs/np.sum(freqs)


    # function to create the random sampling of wind speed and wind directions
    def sampling(self):
        idx = np.random.choice(np.arange(self.wd_array.size),self.samps,p=self.freqs_norm)
        wd = self.wd_array[idx]
        A = self.As[idx]
        k = self.ks[idx]
        ws = A * np.random.weibull(k)
        return wd, ws

    # aep function
    def aep_func(self,x,y):
        aep = np.sum(self.wfm(x,y,wd=self.wd_array,ws=self.ws_array,yaw=0,tilt=0).aep_ilk())
        return aep

    # gradient function
    def aep_jac(self,x,y):
        wd,ws = self.sampling()
        jx,jy = self.wfm.aep_gradients(gradient_method=autograd,wrt_arg=['x','y'],x=x,y=y,ws=ws,wd=wd,time=True)
        daep_sgd = np.array([np.atleast_2d(jx),np.atleast_2d(jy)])*1e6
        return daep_sgd


    def run(self,x_initial,y_initial,run_flag=True):

        # aep component
        aep_comp = CostModelComponent(input_keys=['x','y'],n_wt=self.n_wt,cost_function=self.aep_func,objective=True,cost_gradient_function=self.aep_jac,maximize=True)

        # constraints
        min_spacing_m = 2*self.diameter
        constraint_comp = XYBoundaryConstraint(self.boundary, 'polygon')
        constraints = DistanceConstraintAggregation(constraint_comp,self.n_wt,min_spacing_m,self.wfm.windTurbines)

        # driver
        driver = EasySGDDriver(maxiter=self.maxiter,learning_rate=self.learning_rate)
            
        # define topfarm problem
        tf = TopFarmProblem(
            design_vars = {'x':x_initial, 'y':y_initial},
            cost_comp = aep_comp,
            constraints = constraints,
            driver = driver,
            plot_comp = NoPlot(),
            expected_cost = 1
            )
        
        if run_flag:       # run optimization
            tic = time.time()
            cost, state, recorder = tf.optimize()
            toc = time.time()
            print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])
            x_opt = state['x']
            y_opt = state['y']
            aep_opt = -cost
        else:                   # abort optimization
            x_opt = None
            y_opt = None
            aep_opt = None
            print('Initialization of the layout not successful - Optimization aborted')

        return x_opt,y_opt,aep_opt








class LayoutOptimizationSLSQP():

    def __init__(self,
                 n_wt,
                 wfm,
                 wd_array,
                 ws_array,
                 diameter,
                 boundary,
                 maxiter,
                 tol
                 ):
        
        self.n_wt = n_wt
        self.wfm = wfm
        self.wd_array = wd_array
        self.ws_array = ws_array
        self.diameter = diameter
        self.boundary = boundary
        self.maxiter = maxiter
        self.tol = tol


    # aep function
    def aep_func(self,x,y):
        aep_slsqp = self.wfm(x,y,wd=self.wd_array,ws=self.ws_array).aep().sum().values * 1e6
        return aep_slsqp

    # gradient function
    def aep_jac(self,x,y):
        jx, jy = self.wfm.aep_gradients(gradient_method=autograd,wrt_arg=['x','y'],x=x,y=y,ws=self.ws_array,wd=self.wd_array,time=False)
        daep_slsqp = np.array([np.atleast_2d(jx), np.atleast_2d(jy)])*1e6
        return daep_slsqp


    def run(self,x_initial,y_initial,run_flag=True):

        # aep component
        aep_comp = CostModelComponent(input_keys=['x','y'],n_wt=self.n_wt,cost_function=self.aep_func,objective=True,cost_gradient_function=self.aep_jac,maximize=True)

        # constraints
        min_spacing_m = 2*self.diameter
        print(min_spacing_m)
        constraint_comp = XYBoundaryConstraint(self.boundary, 'polygon')
        constraints = DistanceConstraintAggregation(constraint_comp,self.n_wt,min_spacing_m,self.wfm.windTurbines)

        # driver
        driver = EasyScipyOptimizeDriver(maxiter=self.maxiter,tol=self.tol)
            
        # define topfarm problem
        tf = TopFarmProblem(
            design_vars = {'x':x_initial, 'y':y_initial},
            cost_comp = aep_comp,
            constraints = constraints,
            driver = driver,
            plot_comp = NoPlot(),
            expected_cost = 10
            )
        
        if run_flag:       # run optimization
            tic = time.time()
            cost, state, recorder = tf.optimize()
            toc = time.time()
            print('Optimization with SLSQP took: {:.0f}s'.format(toc-tic))
            x_opt = state['x']
            y_opt = state['y']
            aep_opt = -cost
        else:                   # abort optimization
            x_opt = None
            y_opt = None
            aep_opt = None
            print('Initialization of the layout not successful - Optimization aborted')

        return x_opt,y_opt,aep_opt




class LayoutOptimizationRS():

    def __init__(self,
                 n_wt,
                 wfm,
                 wd_array,
                 ws_array,
                 diameter,
                 boundary,
                 maxiter,
                 maxtime,
                 maxstep
                 ):
        
        self.n_wt = n_wt
        self.wfm = wfm
        self.wd_array = wd_array
        self.ws_array = ws_array
        self.diameter = diameter
        self.boundary = boundary
        self.maxiter = maxiter
        self.maxtime = maxtime
        self.maxstep = maxstep


    # aep function
    def aep_func(self,x,y):
        aep_slsqp = self.wfm(x,y,wd=self.wd_array,ws=self.ws_array).aep().sum().values * 1e6
        return aep_slsqp

    # gradient function
    def aep_jac(self,x,y):
        jx, jy = self.wfm.aep_gradients(gradient_method=autograd,wrt_arg=['x','y'],x=x,y=y,ws=self.ws_array,wd=self.wd_array,time=False)
        daep_slsqp = np.array([np.atleast_2d(jx), np.atleast_2d(jy)])*1e6
        return daep_slsqp


    def run(self,x_initial,y_initial,run_flag=True):

        # aep component
        aep_comp = CostModelComponent(input_keys=['x','y'],n_wt=self.n_wt,cost_function=self.aep_func,objective=True,cost_gradient_function=self.aep_jac,maximize=True)

        # constraints
        min_spacing_m = 2*self.diameter
        print(min_spacing_m)
        constraint_comp = XYBoundaryConstraint(self.boundary, 'polygon')
        constraints = DistanceConstraintAggregation(constraint_comp,self.n_wt,min_spacing_m,self.wfm.windTurbines)

        # driver
        driver = EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(max_step=self.maxstep),max_iter=self.maxiter,max_time=self.maxtime)
            
        # define topfarm problem
        tf = TopFarmProblem(
            design_vars = {'x':x_initial, 'y':y_initial},
            cost_comp = aep_comp,
            constraints = constraints,
            driver = driver,
            plot_comp = NoPlot(),
            expected_cost = 1
            )
        
        if run_flag:       # run optimization
            tic = time.time()
            cost, state, recorder = tf.optimize()
            toc = time.time()
            print('Optimization with RS took: {:.0f}s'.format(toc-tic))
            x_opt = state['x']
            y_opt = state['y']
            aep_opt = -cost
        else:                   # abort optimization
            x_opt = None
            y_opt = None
            aep_opt = None
            print('Initialization of the layout not successful - Optimization aborted')

        return x_opt,y_opt,aep_opt




