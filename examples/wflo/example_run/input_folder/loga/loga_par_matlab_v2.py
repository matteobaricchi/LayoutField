# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:59:31 2024

@author: matteobaricchi
"""

# import packages
import numpy as np
import time
import random
from pathos.multiprocessing import ProcessPool
from functools import partial



class Boundaries():
    
    def __init__(
            self,
            x_boundaries_polygon_list,
            y_boundaries_polygon_list,
            convex_list = False,
            exclusion_list = False
            ):
        """
        Store the information about the boundaries (inclusion/exclusion zones)

        Parameters
        ----------
        x_boundaries_polygon_list : list of array of float64
            list of x coordinates of the polygons that define the domain (both inclusion/exclusion zones).
        y_boundaries_polygon_list : list of array of float64
            list of y coordinates of the polygons that define the domain (both inclusion/exclusion zones).
        convex_list: list bool
            boolean values that refer to the polygon boundaries. True: convex polygon, False: non-convex polygon.
        exclusion_list: list bool
            boolean values that refer to the polygon boundaries. True: exclusion zone, False: inclusion zone.

        Returns
        -------
        None.

        """
        
        if type(x_boundaries_polygon_list)!=list:
            self.x_boundaries_polygon_list = [x_boundaries_polygon_list]
        else:
            self.x_boundaries_polygon_list = x_boundaries_polygon_list
            
        if type(y_boundaries_polygon_list)!=list:
            self.y_boundaries_polygon_list = [y_boundaries_polygon_list]
        else:
            self.y_boundaries_polygon_list = y_boundaries_polygon_list
                    
        self.n_polygons = len(x_boundaries_polygon_list)
        
        if type(convex_list)!=list:
            self.convex_list = [convex_list]*self.n_polygons
        else:
            self.convex_list = convex_list
            
        if type(exclusion_list)!=list:
            self.exclusion_list = [exclusion_list]*self.n_polygons
        else:
            self.exclusion_list = exclusion_list
            
            
    def find_extremes(self):
        
        """
        Determine the x,y extreme values of the polygons that define the boundaries

        Returns
        -------
        x_min : float
            Min x value.
        x_max : float
            Max x value.
        y_min : float
            Min y value.
        y_max : flot
            Max y value.

        """
        
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        
        for i in np.arange(self.n_polygons):
            x_min = np.min([x_min,np.min(self.x_boundaries_polygon_list[i])])
            x_max = np.max([x_max,np.max(self.x_boundaries_polygon_list[i])])
            y_min = np.min([y_min,np.min(self.y_boundaries_polygon_list[i])])
            y_max = np.max([y_max,np.max(self.y_boundaries_polygon_list[i])])
            
        return x_min,x_max,y_min,y_max
            
        


    def enforce_boundaries_MultiPolygon(self,x_mat,y_mat):
        
        """
        This function constarints the points to the boundaries defined by the polygon 
        received as input (both inclusion/exclusion zone). It allows multiple polygons
        as inclusion and exclusion zones.
        
        n_pop indicates the size of the population.
        n_wt indicates the number of wind turbines.
        n_segments indicates the number of the segments of the polygon.

        The first coordinates of the polygons must be repeated to ensure that is closed shape.

        Parameters
        ----------
        x_mat : array of float64, size : (n_pop,n_wt)
            x coordinates of the layouts.
        y_mat : array of float64, size : (n_pop,n_wt)
            y coordinates of the layouts.

        Returns
        -------
        x_mat_new :  array of float64, size : (n_pop,n_wt)
            x coordinates of the constrained layouts.
        y_mat_new :  array of float64, size : (n_pop,n_wt)
            y coordinates of the constrained layouts.

        """
        
        def _distance_from_boundaries(x_A,y_A,x_B,y_B,x_P,y_P):
            
            """
            This function is used by the enforce_boundaries function to calculate the 
            distance from the boundaries and the nearest point on the boundaries.
            
            n_pop indicates the size of the population.
            n_wt indicates the number of wind turbines.
            n_segments indicates the number of the segments of the polygon.

            Parameters
            ----------
            x_A : array of float64, size: (n_pop,n_wt,n_segments)
                x coordinates of the first vertices of the segments of the polygon.
            y_A : array of float64, size: (n_pop,n_wt,n_segments)
                y coordinates of the first vertices of the segments of the polygon.
            x_B : array of float64, size: (n_pop,n_wt,n_segments)
                x coordinates of the second vertices of the segments of the polygon.
            y_B : array of float64, size: (n_pop,n_wt,n_segments)
                y coordinates of the second vertices of the segments of the polygon.
            x_P : array of float64, size: (n_pop,n_wt,n_segments)
                x coordinates of the wind turbines.
            y_P : array of float64, size: (n_pop,n_wt,n_segments)
                y coordinates of the wind turbines.

            Returns
            -------
            d : array of float64, size: (n_pop,n_wt,n_segments)
                distance between the wind turbines and the line obtained from the segments.
            x_nearest : array of float64, size: (n_pop,n_wt,n_segments)
                x coordinates of the nearest point on the segment to the turbine.
            y_nearest : array of float64, size: (n_pop,n_wt,n_segments)
                y coordinates of the nearest point on the segment to the turbine.

            """
            
            # P: point of interest
            # A: first point segment
            # B: second point segment
            # K: projection of P on the segment
            # W: closest point to the segment

            # case 1: A is the closest point to P
            # case 2: B is the closest point to P
            # case 3: K is the closest point to P
            fil_1 = (x_B-x_A)*(x_P-x_A)+(y_B-y_A)*(y_P-y_A)<0
            fil_2 = (x_B-x_A)*(x_P-x_B)+(y_B-y_A)*(y_P-y_B)>0
            fil_3 = np.logical_not(fil_1|fil_2)

            # distance from the line of the segment: d>=0 => on the right of the segment (clockwise direction)
            d = (((x_P-x_A)*(y_B-y_A)+(y_P-y_A)*(-x_B+x_A))/np.sqrt((x_B-x_A)**2+(y_B-y_A)**2))

            # cakculate projected point on the segment
            x_K = x_P-(((x_P-x_A)*(y_B-y_A)+(y_P-y_A)*(-x_B+x_A))/((x_B-x_A)**2+(y_B-y_A)**2))*(y_B-y_A)
            y_K = y_P-(((x_P-x_A)*(y_B-y_A)+(y_P-y_A)*(-x_B+x_A))/((x_B-x_A)**2+(y_B-y_A)**2))*(-x_B+x_A)

            # initialize output 
            x_nearest = np.zeros_like(x_P)
            y_nearest = np.zeros_like(y_P)
            
            # calculate nearest point on the boundaries
            x_nearest[fil_1] = x_A[fil_1]
            y_nearest[fil_1] = y_A[fil_1]
            x_nearest[fil_2] = x_B[fil_2]
            y_nearest[fil_2] = y_B[fil_2]
            x_nearest[fil_3] = x_K[fil_3]
            y_nearest[fil_3] = y_K[fil_3]
            
            return d,x_nearest,y_nearest


        def _is_inside_ray_casting(x_A,y_A,x_B,y_B,x_P,y_P):
            
            """
            This function determines wheter a point is inside a polygon through method 
            based on ray-casting to be effective also for non-convex polygons.
            
            n_pop indicates the size of the population.
            n_wt indicates the number of wind turbines.
            n_segments indicates the number of the segments of the polygon.

            Parameters
            ----------
            x_A : array of float64, size: (n_pop,n_wt,n_segments)
                x coordinates of the first vertices of the segments of the polygon.
            y_A : array of float64, size: (n_pop,n_wt,n_segments)
                y coordinates of the first vertices of the segments of the polygon.
            x_B : array of float64, size: (n_pop,n_wt,n_segments)
                x coordinates of the second vertices of the segments of the polygon.
            y_B : array of float64, size: (n_pop,n_wt,n_segments)
                y coordinates of the second vertices of the segments of the polygon.
            x_P : array of float64, size: (n_pop,n_wt,n_segments)
                x coordinates of the wind turbines.
            y_P : array of float64, size: (n_pop,n_wt,n_segments)
                y coordinates of the wind turbines.

            Returns
            -------
            is_inside : array of bool, size: (n_pop,n_wt)
                True: the point is inside the polygon, False: the point is outside the polygon.

            """
            
            # P: point of interest
            # A: first point segment
            # B: second point segment
            # Q: intersection segment-ray

            # calculate intersection segment-ray
            x_Q = (y_P-y_A)*((x_B-x_A)/(y_B-y_A))+x_A

            # check if the intersection occurs outside the segment
            fil_outside = ((x_Q<x_A) & (x_Q<x_B)) | ((x_Q>x_A) & (x_Q>x_B))
            x_Q[fil_outside] = np.inf

            # sort segments based on x_Q
            x_Q_sorted = np.sort(x_Q,axis=2)

            # find the position of x_P within x_Q and extract index
            x_Q_on_the_left = x_Q_sorted<x_P
            ind_x_P_sorted = np.sum(x_Q_on_the_left,axis=2)

            # index of x_P: odd=inside, even=outside
            is_inside = ind_x_P_sorted%2!=0
            
            return is_inside

            
        # initialize variables
        allowed_inclusion = np.zeros_like(x_mat,dtype='bool')
        allowed_exclusion = np.ones_like(x_mat,dtype='bool')
        x_nearest = np.ones_like(x_mat)*np.inf
        y_nearest = np.ones_like(y_mat)*np.inf
        d_nearest = np.ones_like(x_mat)*np.inf
        
        # iterate for each polygon
        for i in np.arange(self.n_polygons):
            
            # define variables
            x_boundaries_polygon = self.x_boundaries_polygon_list[i]
            y_boundaries_polygon = self.y_boundaries_polygon_list[i]
            convex = self.convex_list[i]
            exclusion = self.exclusion_list[i]
        
            # define extremes of each segment of the domain (clockwise direction)
            x_A = x_boundaries_polygon[:-1]
            y_A = y_boundaries_polygon[:-1]
            x_B = x_boundaries_polygon[1:]
            y_B = y_boundaries_polygon[1:]
            n_segments = len(x_A)
        
            # extend the size of the matrices
            n_pop = x_mat.shape[0]
            n_wt = x_mat.shape[1]
            x_mat_ext = np.tile(np.reshape(x_mat,(n_pop,n_wt,1)),(1,1,n_segments))
            y_mat_ext = np.tile(np.reshape(y_mat,(n_pop,n_wt,1)),(1,1,n_segments))
            x_A_ext = np.tile(np.reshape(x_A,(1,1,n_segments)),(n_pop,n_wt,1))
            y_A_ext = np.tile(np.reshape(y_A,(1,1,n_segments)),(n_pop,n_wt,1))
            x_B_ext = np.tile(np.reshape(x_B,(1,1,n_segments)),(n_pop,n_wt,1))
            y_B_ext = np.tile(np.reshape(y_B,(1,1,n_segments)),(n_pop,n_wt,1))
        
            # calculate distance from each segment
            d_ext,x_nearest_ext,y_nearest_ext = _distance_from_boundaries(x_A_ext,y_A_ext,x_B_ext,y_B_ext,x_mat_ext,y_mat_ext)
        
            # check if the points are within the boundaries defined by the polygon (clockwise direction)
            if convex:
                is_inside_ext = d_ext>=0
                is_inside = np.all(is_inside_ext,axis=2)
            else:
                is_inside = _is_inside_ray_casting(x_A_ext,y_A_ext,x_B_ext,y_B_ext,x_mat_ext,y_mat_ext)
        
            # find x,y of the nearest points on the boundaries
            d_nearest_ext = np.sqrt((x_mat_ext-x_nearest_ext)**2+(y_mat_ext-y_nearest_ext)**2)
            ind_nearest = np.argmin(d_nearest_ext,axis=2)
            ind_n_pop,ind_n_wt = np.indices((n_pop,n_wt))
            x_nearest_temp = x_nearest_ext[ind_n_pop,ind_n_wt,ind_nearest]
            y_nearest_temp = y_nearest_ext[ind_n_pop,ind_n_wt,ind_nearest]
            d_nearest_temp = d_nearest_ext[ind_n_pop,ind_n_wt,ind_nearest]
            
            # update allowed variable (inclusion and exclusion zones are addressed separately)
            if exclusion:
                allowed_exclusion = allowed_exclusion & (~is_inside)
            else:
                allowed_inclusion = allowed_inclusion | is_inside

            # update x_nearest,y_nearest variables
            x_nearest[d_nearest_temp<d_nearest] = x_nearest_temp[d_nearest_temp<d_nearest]
            y_nearest[d_nearest_temp<d_nearest] = y_nearest_temp[d_nearest_temp<d_nearest]
            d_nearest[d_nearest_temp<d_nearest] = d_nearest_temp[d_nearest_temp<d_nearest]
        
        
        # calculate allowed variable
        allowed = allowed_inclusion & allowed_exclusion
            
        # enforce boundaries 
        x_mat_new = x_mat.copy()
        y_mat_new = y_mat.copy()
        x_mat_new[~allowed] = x_nearest[~allowed]
        y_mat_new[~allowed] = y_nearest[~allowed]

        return x_mat_new,y_mat_new
    




class LayoutOptimizationGA():
    
    def __init__(
            self,
            fitness_function,
            n_wt,
            boundaries,
            n_gen,
            n_pop,
            n_keep_parents = 3,
            perc_s = 0.7,
            perc_m = 0.3,
            step_m = 'default',
            distance_limit = 'default',
            x_mat_initial = None,
            y_mat_initial = None,
            parallel_execution = False,
            n_cpu = None,
            full_pop_evaluation = False,
            gen_to_f = False
            ):
        
        """
        Parameters
        ----------
        fitness_function : function
            fitness function to maximize which requires as input: (x,y) coordinates.
        boundaries : Boundary object
            contains the information about the boundaries.
        n_wt : int
            number of wind turbines.
        n_gen : int
            number of generations.
        n_pop : int
            population size.
        n_keep_parents : int
            number of parent solutions kept in the next generation.
            
        p_s_array : array of float64, size : (n_gen,)
            values of percentage of selection for each generation.
        p_m_array : array of float64, size : (n_gen,)
            values of percentage of mutation for each generation.
        s_m_array : array of float64, size : (n_gen,)
            values of step of mutation for each generation.
            
        distance_limit : float
            max distance between two tubrines to be identified as paired (required for the crossover).
            
            
        x_mat_initial : array of float64
            x coordinates of the initial population.
        y_mat_initial : array of float64
            y coordinates of the initial population.

            
        """       
        
        
        def _regular_layout(n_wt,p_density,p_rated):
            
            """
            This function creates a regular layout with aligned turbines.
    
            Parameters
            ----------
            n_wt : int
                number of wind turbines.
            p_density : float
                Power density expressed in W/m2.
            p_rated : float
                Rated power of the indiviudal turbine.
    
            Returns
            -------
            x : array of float64, size : (n_wt,)
                x coordinates of the turbines.
            y : array of float64, size : (n_wt,)
                y coordinates of the turbines.
            """
    
            # define number of row and columns and calculate area and side
            n_row = int(np.floor(np.sqrt(n_wt)))
            n_col = int(np.ceil(n_wt/n_row))
            A = (n_wt*p_rated/p_density)*10**6
            l = np.sqrt(A)
            
            # create x,y 
            x_vec = np.linspace(0,1,n_col)*l
            y_vec = np.linspace(0,1,n_row)*l
            x_mat = np.tile(np.reshape(x_vec,(1,len(x_vec))),(len(y_vec),1))
            y_mat = np.tile(np.reshape(y_vec,(len(y_vec),1)),(1,len(x_vec)))
    
            # fix dimentions of x,y
            x_temp = x_mat.reshape(-1)
            y_temp = y_mat.reshape(-1)
            x = x_temp[0:n_wt]
            y = y_temp[0:n_wt]
            
            return x,y
        
        
        
        def _dynamic_mutatation_paramters(n_gen,p_m_min,p_m_max,s_m_min,s_m_max,p_gen_start,p_gen_stop):
            
            """
            This function is used to generate the mutation parameters enabling dynamic mutation.
            Working principle: the percentage of mutation and the muataion step are gradually decreased (linearly) along the generations.
    
            Parameters
            ----------
            n_gen : int
                number of generations.
            p_m_min : float
                min value for the percentage of mutation (p_m).
            p_m_max : float
                max value for the percentage of mutation (p_m).
            s_m_min : float
                min value for the step of mutation (s_m).
            s_m_max : float
                max value for the step of mutation (s_m).
            p_gen_start : int
                number of generation at which the p_m and s_m start decreasing.
            p_gen_stop : int
                number of generation at which the p_m and s_m stop decreasing.
    
            Returns
            -------
            p_m : array of float64, size : (n_gen,)
                values of percentage of mutation for each generation.
            s_m : array of float64, size : (n_gen,)
                values of step of mutation for each generation.
    
            """
            
            p_m = np.array([p_m_max]*n_gen)
            s_m = np.array([s_m_max]*n_gen)
            
            gen_start = int(np.floor(p_gen_start*n_gen))
            gen_stop = int(np.floor(p_gen_stop*n_gen))
            
            p_m[gen_start:gen_stop] = np.linspace(p_m_max,p_m_min,gen_stop-gen_start)
            s_m[gen_start:gen_stop] = np.linspace(s_m_max,s_m_min,gen_stop-gen_start)
            
            p_m[gen_stop:] = p_m_min
            s_m[gen_stop:] = s_m_min
            
            return p_m,s_m
    
    
    
    
    
        def _dynamic_selection_paramters(n_gen,p_s_min,p_s_max,p_gen_start,p_gen_stop):
            
            """
            This function is used to generate the selection parameters enabling dynamic selection.
            Working principle: the percentage of selection is gradually decreased (linearly) along the generations.
    
            Parameters
            ----------
            n_gen : int
                number of generations.
            p_s_min : float
                min value for the percentage of selection (p_s).
            p_s_max : float
                max value for the percentage of selection (p_s).
            p_gen_start : int
                number of generation at which the p_s start increasing.
            p_gen_stop : int
                number of generation at which the p_s stop increasing.
    
            Returns
            -------
            p_s : array of float64, size : (n_gen,)
                values of percentage of selection for each generation.
    
            """
            
            p_s = np.array([p_s_min]*n_gen)
            
            gen_start = int(np.floor(p_gen_start*n_gen))
            gen_stop = int(np.floor(p_gen_stop*n_gen))
            
            p_s[gen_start:gen_stop] = np.linspace(p_s_min,p_s_max,gen_stop-gen_start)
            
            p_s[gen_stop:] = p_s_max
            
            return p_s


        
        # store fitness function
        self.fitness_function = fitness_function
        
        # store number of turbines
        self.n_wt = n_wt
        
        # create and store boundaries object
        self.boundaries = boundaries
        
        # store optimzation parameters
        self.n_gen = n_gen
        self.n_pop = n_pop
        self.n_keep_parents = n_keep_parents
        
        # store parallel execution parameters
        self.parallel_execution = parallel_execution
        self.n_cpu = n_cpu
        self.full_pop_evaluation = full_pop_evaluation
        
        
        # create selection parameters array
        
        if (type(perc_s)==float) or (type(perc_s)==np.float64):
            # use the same selection parameters for every generation
            self.p_s_array = np.ones(n_gen)*perc_s
            
        elif len(perc_s)==2:
            # create selection parameters array based on min and max values
            p_s_min = np.min(perc_s)
            p_s_max = np.max(perc_s)
            self.p_s_array = _dynamic_selection_paramters(self.n_gen,p_s_min,p_s_max,p_gen_start=0,p_gen_stop=1)
            
        elif len(perc_s)==self.n_gen:
            # create selection parameters array based on the input array
            self.p_s_array = perc_s
            
        else:
            raise ValueError("Incorrect size of 'perc_s'.")

        
        # create mutation parameters array
        
        if type(step_m)==str:
            
            # check if the default value is selected fot the step of mutation and calcualte it
            if step_m=='default':
                x_min,x_max,y_min,y_max = boundaries.find_extremes()
                step_m = float(np.sqrt((x_max-x_min)*(y_max-y_min))/n_wt)
                
            else:
                raise ValueError("Incorrect values or size of 'perc_m' and 'step_m.")
        
        if ((type(perc_m)==float) or (type(perc_m)==np.float64)) and ((type(step_m)==float) or (type(step_m)==np.float64)):
            # use the same mutation parameters for every generation
            self.p_m_array = np.ones(n_gen)*perc_m
            self.s_m_array = np.ones(n_gen)*step_m
            
        elif (len(perc_m)==2) and (len(step_m)==2):
            # create mutation parameters array based on min and max values
            p_m_min = np.min(perc_m)
            p_m_max = np.max(perc_m)
            s_m_min = np.min(step_m)
            s_m_max = np.max(step_m)
            self.p_m_array,self.s_m_array = _dynamic_mutatation_paramters(self.n_gen,p_m_min,p_m_max,s_m_min,s_m_max,p_gen_start=0,p_gen_stop=1)
            
        elif (len(perc_m)==self.n_gen) and (len(step_m)==self.n_gen):
            # create mutation parameters array based on the input array
            self.p_m_array = perc_m
            self.s_m_array = step_m
            
        else:
            raise ValueError("Incorrect values or size of 'perc_m' and 'step_m.")
            


        # define parameter for turbine association
        
        if type(distance_limit)==str:
            
            if distance_limit=='default':
                # calculate and assign a realistic value
                x_min,x_max,y_min,y_max = boundaries.find_extremes()
                self.d_limit = np.sqrt((x_max-x_min)*(y_max-y_min))/n_wt
                
            else:
                raise ValueError("Incorrect value of 'distance_limit.")
         
        elif (type(distance_limit)==float or (type(distance_limit)==int)) or (type(distance_limit)==np.float64):
            # assign input value
            self.d_limit = distance_limit
            
        else:
            raise ValueError("Incorrect value of 'distance_limit.")

        





        # create initial population matrices
        
        if ((x_mat_initial is not None) and (y_mat_initial is not None)):
            
            # check that x_mat_initial and y_mat_initial have the same shape
            if x_mat_initial.shape==y_mat_initial.shape:
                
                # matrices already extended to match the population size
                if len(x_mat_initial.shape)==2:
            
                    # case: size (n_pop,n_wt)
                    if (x_mat_initial.shape[0]==self.n_pop) and (x_mat_initial.shape[1]==self.n_wt):
                        self.x_mat_initial = x_mat_initial
                        self.y_mat_initial = y_mat_initial
                        
                    # case: size (n_wt,n_pop)
                    elif (x_mat_initial.shape[0]==self.n_pop) and (x_mat_initial.shape[1]==self.n_wt):
                        self.x_mat_initial = np.transpose(x_mat_initial)
                        self.y_mat_initial = np.transpose(y_mat_initial)
                        
                    else:
                        raise ValueError("Incorrect size for 'x_mat_initial' and 'y_mat_initial'. Their size must be: (n_pop,n_wt), (n_wt,n_pop) or (n_wt,)")
                
                # matrices need to be extend to match the population size
                elif len(x_mat_initial.shape)==1:
                    
                    # case: size (n_wt,)
                    if x_mat_initial.shape[0]==self.n_wt:
                        self.x_mat_initial = np.tile(np.reshape(x_mat_initial,(1,self.n_wt)),(self.n_pop,1))
                        self.y_mat_initial = np.tile(np.reshape(y_mat_initial,(1,self.n_wt)),(self.n_pop,1))
                        
                    else:
                        raise ValueError("Incorrect size for 'x_mat_initial' and 'y_mat_initial'. Their size must be: (n_pop,n_wt), (n_wt,n_pop) or (n_wt,)")
                            
                else:
                    raise ValueError("Incorrect size for 'x_mat_initial' and 'y_mat_initial'. Their size must be: (n_pop,n_wt), (n_wt,n_pop) or (n_wt,)")

            else:
                raise ValueError("Incorrect size for 'x_mat_initial' and 'y_mat_initial'. They must have the same size.")
            
        else:
            
            # generate initial population based on n_wt
                        
            # create a regular layout
            x_min,x_max,y_min,y_max = boundaries.find_extremes()
            p_rated = 10 # not relevant
            p_density = (p_rated*self.n_wt*1e6)/((x_max-x_min)*(y_max-y_min))
            x_reg_temp,y_reg_temp = _regular_layout(n_wt=self.n_wt,p_density=p_density,p_rated=p_rated)
            x_reg = x_reg_temp+x_min
            y_reg = y_reg_temp+y_min
        
            self.x_mat_initial = np.tile(np.reshape(x_reg,(1,self.n_wt)),(self.n_pop,1))
            self.y_mat_initial = np.tile(np.reshape(y_reg,(1,self.n_wt)),(self.n_pop,1))
        
        
        # enforce boundaries
        self.x_mat_initial,self.y_mat_initial = self.boundaries.enforce_boundaries_MultiPolygon(self.x_mat_initial,self.y_mat_initial)

   
        # initialize output values
        self.x_opt = None
        self.y_opt = None
        self.fitness_val_opt = None
        self.fitness_val_mat_opt = None
        self.fitness_val_array_gen = None

        # store gen_to_f
        self.gen_to_f = gen_to_f



    def _mutation_layoutOpt(self,x_mat,y_mat,fitness_val_mat,s_m,p_m):
        
        """
        This function applies a mutation to a layout identified by x,y arrays.
        The function is vectorized to compute the operation for all the population.
        The size of the population is identified by n_pop (not required as parameter).

        Parameters
        ----------
        x_mat : array of float64, size : (n_pop,n_wt)
            x coordinates of the turbines.
        y_mat : array of float64, size : (n_pop,n_wt)
            y coordinates of the turbines.
        fitness_val_mat : array of float64, size : (n_pop,n_wt)
            value of the fitness function to maximize for each turbine.
        s_m : float
            random step of the mutation.
        p_m : float
            percentage of mutation for an individual turbine (e.g. p_m=0.1 --> 1 turbine out of 10 is relocated), expressed as a fraction (i.e. p_m=0.1 --> 10%).

        Returns
        -------
        x_mat_new : array of float64, size : (n_pop,n_wt)
            x coordinates of the turbines after the mutation.
        y_mat_new : array of float64, size : (n_pop,n_wt)
            y coordinates of the turbines after the mutation.
        """
        
        # rank the turbines from higher to lower fitness
        ind_ranked = np.argsort(-fitness_val_mat,axis=1)
        
        # create probability matrix (LINEAR INCREASE)
        p_val_array = np.minimum(self.n_wt*p_m*np.linspace(0,1,self.n_wt)/np.sum(np.linspace(0,1,self.n_wt)),1)
        p_val_mat = np.tile(np.reshape(p_val_array,(1,len(p_val_array))),(self.n_pop,1))
        
        # assign probability values to each turbine (high probability for low fitness value -> these turbines will mutate)
        ind_col = ind_ranked.reshape(-1)
        ind_row = np.tile(np.reshape(np.arange(0,self.n_pop),(self.n_pop,1)),(1,self.n_wt)).reshape(-1)
        p_mat_temp = np.zeros((self.n_pop,self.n_wt))
        p_mat_temp[ind_row,ind_col] = p_val_mat.reshape(-1)
        p_mat = np.reshape(p_mat_temp,(self.n_pop,self.n_wt))
        
        # create mutation bool matrix (0 = no mutation, 1 = mutation)
        mutation_bool_mat = np.random.rand(self.n_pop,self.n_wt)<p_mat
        mutation_int_mat = np.zeros((self.n_pop,self.n_wt))
        mutation_int_mat[mutation_bool_mat] = 1
        
        # generate random step and direction
        step = s_m*np.random.rand(self.n_pop,self.n_wt)
        direction = ((2*np.pi))*np.random.rand(self.n_pop,self.n_wt)
        
        # apply mutation
        x_mat_new = x_mat+mutation_int_mat*step*np.cos(direction)
        y_mat_new = y_mat+mutation_int_mat*step*np.sin(direction)
        
        return x_mat_new,y_mat_new



    def _selection_layoutOpt(self,x_mat,y_mat,fitness_val_mat,p_s):
        
        """
        This function applies a selection to population of layouts based on the "tournament technique".
        It includes a preliminary selection based on the percentage of selection.
        The function is vectorized to compute the operation for all the population.
        The size of the population is identified by n_pop (not required as parameter).

        Parameters
        ----------
        x_mat : array of float64, size : (n_pop,n_wt)
            x coordinates of the turbines.
        y_mat : array of float64, size : (n_pop,n_wt)
            y coordinates of the turbines.
        fitness_val_mat : array of float64, size : (n_pop,n_wt)
            value of the fitness function to maximize for each turbine.
        p_s : float
            percentage of selection, expressed as a fraction (i.e. p_s=0.1 --> 10%).

        Returns
        -------
        x_mat_parents : array of float64, size : (n_pop,n_wt)
            x coordinates of the turbines after the selection.
        y_mat_parents : array of float64, size : (n_pop,n_wt)
            y coordinates of the turbines after the selection.
        fitness_val_mat_parents : array of float64, size : (n_pop,n_wt)
            fitness values of the parent population.

        """

        # calculate the fitness of each solution in the population
        fitness_val_array = np.sum(fitness_val_mat,axis=1)
        
        # PRELIMINARY STAGE: sort the population and filter the population based on the percentage of selection
        ind_sorted = np.argsort(fitness_val_array)
        ind_filtered = ind_sorted[int(np.floor((1-p_s)*self.n_pop)):]
        
        # complete and shuffle the selected indices
        ind_tournament_A = np.concatenate((ind_filtered,np.random.choice(ind_filtered,size=self.n_pop-len(ind_filtered))))
        ind_tournament_B = np.concatenate((ind_filtered,np.random.choice(ind_filtered,size=self.n_pop-len(ind_filtered))))
        random.shuffle(ind_tournament_A)
        random.shuffle(ind_tournament_B)
        
        # play tournament
        ind_parents = ind_tournament_B
        cond_winner_A = fitness_val_array[ind_tournament_A]>fitness_val_array[ind_tournament_B]
        ind_parents[cond_winner_A] = ind_tournament_A[cond_winner_A]
        
        # create output
        fitness_val_mat_parents = fitness_val_mat[ind_parents,:]
        x_mat_parents = x_mat[ind_parents,:]
        y_mat_parents = y_mat[ind_parents,:]
        
        return x_mat_parents,y_mat_parents,fitness_val_mat_parents






    def _crossover_layoutOpt(self,x_mat_parents,y_mat_parents,fitness_val_mat_parents):
        
        """
        This function iterates the crossover over the entire population.

        Parameters
        ----------
        x_mat_parents : array of float64, size : (n_pop,n_wt)
            x coordinates of the parent layouts.
        y_mat_parents : array of float64, size : (n_pop,n_wt)
            y coordinates of the parent layouts.
        fitness_val_mat_parents : array of float64, size : (n_pop,n_wt)
            fitness values of the parent layouts.

        Returns
        -------
        x_mat_children : array of float64, size : (n_pop,n_wt)
            x coordinates of the children layouts.
        y_mat_children : array of float64, size : (n_pop,n_wt)
            y coordinates of the parent layouts.

        """

        def _crossover_function(x_parent_1,x_parent_2,y_parent_1,y_parent_2,f_parent_1,f_parent_2):
            
            """
            This function applies the crossover between two parent solutions (layouts), combining linear and random crossover.
            It can only be applied to two solutions (layouts) individually.
            Working principle: each turbine is identified as paired or outliers, for the 
            former linear crossover is applied while for the latter random crossover is applied.

            Parameters
            ----------
            x_parent_1 : array of float64, size: (n_wt,)
                x coordinates of parent layout 1.
            x_parent_2 : array of float64, size: (n_wt,)
                x coordinates of parent layout 2.
            y_parent_1 : array of float64, size: (n_wt,)
                y coordinates of parent layout 1.
            y_parent_2 : array of float64, size: (n_wt,)
                y coordinates of parent layout 2.
            f_parent_1 : array of float64, size: (n_wt,)
                fitness values of parent layout 1.
            f_parent_2 : array of float64, size: (n_wt,)
                fitness values of parent layout 2.

            Returns
            -------
            x_child_1 : array of float64, size: (n_wt,)
                x coordinates of child layout 1.
            x_child_2 : array of float64, size: (n_wt,)
                x coordinates of child layout 2.
            y_child_1 : array of float64, size: (n_wt,)
                y coordinates of child layout 1.
            y_child_2 : array of float64, size: (n_wt,)
                y coordinates of child layout 2.

            """
            
            def _associate_turbines(x_parent_1,x_parent_2,y_parent_1,y_parent_2):
                
                """
                This function identifies which turbines occupy the same location (within a radius equal to d_limit) between two different layouts.
                It can only be applied to two solutions (layouts) individually.
                Working principle: for each turbine of the layout 1, the closest turbine of layout 2 is associated if its distance is lower than d_limit.
                Limitation: if two turbines of layout 1 have the same closest turbine, the first turbine of layout 1 is prioritized and the second turbine is appointed as outlier.

                Parameters
                ----------
                x_parent_1 : array of float64, size : (n_wt,)
                    x coordinates of layout 1.
                x_parent_2 : array of float64, size : (n_wt,)
                    y coordinates of layout 1.
                y_parent_1 : array of float64, size : (n_wt,)
                    x coordinates of layout 2.
                y_parent_2 : array of float64, size : (n_wt,)
                    y coordinates of layout 2.

                Returns
                -------
                ind_paired_parent_1 : array of int, size: (number of paired turbines,)
                    indices of the paired turbines of layout 1 with layout 2 (ordered).
                ind_paired_parent_2 : array of int, size: (number of paired turbines,)
                    indices of the paired turbines of layout 2 with layout 1 (ordered).
                ind_outliers_parent_1 : array of int, size: (number of outlier turbines,)
                    indices of the outlier turbines of layout 1.
                ind_outliers_parent_2 : array of int, size: (number of outlier turbines,)
                    indices of the outlier turbines of layout 2.

                """

                n_wt = len(x_parent_1)
                
                # calculate distance
                x_parent_1_ext = np.tile(np.reshape(x_parent_1,(1,n_wt)),(n_wt,1))
                x_parent_2_ext = np.tile(np.reshape(x_parent_2,(n_wt,1)),(1,n_wt))
                y_parent_1_ext = np.tile(np.reshape(y_parent_1,(1,n_wt)),(n_wt,1))
                y_parent_2_ext = np.tile(np.reshape(y_parent_2,(n_wt,1)),(1,n_wt))
                d_mat = np.sqrt((x_parent_1_ext-x_parent_2_ext)**2+(y_parent_1_ext-y_parent_2_ext)**2)
                
                # create mask to identify only the nearest turbine of each turbine of parent 1
                ind_d_min_parent_1 = np.argmin(d_mat,axis=0)
                ind_mat_temp = np.tile(np.reshape(np.arange(0,n_wt),(n_wt,1)),(1,n_wt))
                ind_d_min_parent_1_ext = np.tile(np.reshape(ind_d_min_parent_1,(1,n_wt)),(n_wt,1))
                ind_d_min_parent_1_ext[ind_d_min_parent_1_ext!=ind_mat_temp] = -1
                d_mat_filtered_temp = np.inf*np.ones((n_wt,n_wt))
                d_mat_filtered_temp[ind_d_min_parent_1_ext>=0] = d_mat[ind_d_min_parent_1_ext>=0]
                
                # avoid duplicates
                d_mat_filtered = np.inf*np.ones((n_wt,n_wt))
                d_mat_filtered[np.arange(0,n_wt),np.argmin(d_mat_filtered_temp,axis=1)] = d_mat_filtered_temp[np.arange(0,n_wt),np.argmin(d_mat_filtered_temp,axis=1)]
                
                # obtain the index of the nearest turbine for each turbine of parent 1 avoiding duplicates (-1 indicates outliers)
                ind_d_min_parent_1_filtered = np.argmin(d_mat_filtered,axis=0)
                ind_d_min_parent_1_filtered[np.min(d_mat_filtered,axis=0)>self.d_limit] = -1
                
                # find paired values
                ind_paired_parent_1 = np.arange(0,n_wt)[ind_d_min_parent_1_filtered>=0]
                ind_paired_parent_2 = ind_d_min_parent_1_filtered[ind_d_min_parent_1_filtered>=0]
                
                # find outliers parent 1
                ind_outliers_parent_1 = np.arange(0,n_wt)[ind_d_min_parent_1_filtered<0]
                
                # find outliers parent 2
                ind_mat_temp = np.tile(np.reshape(np.arange(0,n_wt),(n_wt,1)),(1,n_wt))
                ind_d_min_parent_1_filtered_ext = np.tile(np.reshape(ind_d_min_parent_1_filtered,(1,n_wt)),(n_wt,1))
                ind_mat_temp_filtered = -np.ones((n_wt,n_wt),dtype=int)
                ind_mat_temp_filtered[ind_mat_temp==ind_d_min_parent_1_filtered_ext] = ind_d_min_parent_1_filtered_ext[ind_mat_temp==ind_d_min_parent_1_filtered_ext]
                ind_paired_parent_2_ordered_temp = np.max(ind_mat_temp_filtered,axis=1) 
                ind_outliers_parent_2 = np.arange(0,n_wt)[ind_paired_parent_2_ordered_temp<0]
                
                return ind_paired_parent_1,ind_paired_parent_2,ind_outliers_parent_1,ind_outliers_parent_2


            def _linear_crossover(x_1,x_2,y_1,y_2,f_1,f_2):
                
                """
                This function applies the linear crossover between two parent (partial) layouts.
                It can only be applied to two solutions (layouts) individually.
                Working prinicple: both children are obtained along the line that connects the two paired turbines, the first one is in between (depending on f_1,f_2) and the second is extrpolated towards the best turbine
                
                Parameters
                ----------
                x_1 : array of float64, size: (number of paired turbines,)
                    x coordinates of (partial) parent layout 1.
                x_2 : array of float64, size: (number of paired turbines,)
                    x coordinates of (partial) parent layout 2.
                y_1 : array of float64, size: (number of paired turbines,)
                    y coordinates of (partial) parent layout 1.
                y_2 : array of float64, size: (number of paired turbines,)
                    y coordinates of (partial) parent layout 2.
                f_1 : array of float64, size: (number of paired turbines,)
                    fitness function values correspondent to the turbines identified by (x_1,y_1).
                f_2 : array of float64, size: (number of paired turbines,)
                    fitness function values correspondent to the turbines identified by (x_2,y_2).

                Returns
                -------
                x_c1 : array of float64, size: (number of paired turbines,)
                    x coordinates of (partial) child layout 1.
                x_c2 : array of float64, size: (number of paired turbines,)
                    x coordinates of (partial) child layout 2.
                y_c1 : array of float64, size: (number of paired turbines,)
                    y coordinates of (partial) child layout 1.
                y_c2 : array of float64, size: (number of paired turbines,)
                    y coordinates of (partial) child layout 2.
                    
                """
                
                # avoid condition f_1=f_2=0
                check = (f_1==0) & (f_2==0)
                f_1[check] = 1
                f_2[check] = 1
                
                # point between the turbines
                x_c1 = (f_1/(f_1+f_2))*x_1+(f_2/(f_1+f_2))*x_2
                y_c1 = (f_1/(f_1+f_2))*y_1+(f_2/(f_1+f_2))*y_2
                
                # point on the side of the better turbine
                x_c2_temp_1 = x_1+(x_1-x_2)*(f_1/(f_1+f_2))
                y_c2_temp_1 = y_1+(y_1-y_2)*(f_1/(f_1+f_2))
                x_c2_temp_2 = x_2+(x_2-x_1)*(f_2/(f_1+f_2))
                y_c2_temp_2 = y_2+(y_2-y_1)*(f_2/(f_1+f_2))
                x_c2 = x_c2_temp_1
                x_c2[f_2>f_1] = x_c2_temp_2[f_2>f_1]
                y_c2 = y_c2_temp_1
                y_c2[f_2>f_1] = y_c2_temp_2[f_2>f_1]
                
                return x_c1,x_c2,y_c1,y_c2


            def _random_crossover(x_1,x_2,y_1,y_2):
                
                """
                This function applies the random crossover between two parent (partial) layouts.
                It can only be applied to two solutions (layouts) individually.
                Working principle: the turbine of the children are selected randomly between the parent layouts.

                Parameters
                ----------
                x_1 : array of float64, size: (number of outlier turbines,)
                    x coordinates of (partial) parent layout 1.
                x_2 : array of float64, size: (number of outlier turbines,)
                    x coordinates of (partial) parent layout 2.
                y_1 : array of float64, size: (number of outlier turbines,)
                    y coordinates of (partial) parent layout 1.
                y_2 : array of float64, size: (number of outlier turbines,)
                    y coordinates of (partial) parent layout 2.

                Returns
                -------
                x_c1 : array of float64, size: (number of outlier turbines,)
                    x coordinates of (partial) child layout 1.
                x_c2 : array of float64, size: (number of outlier turbines,)
                    x coordinates of (partial) child layout 2.
                y_c1 : array of float64, size: (number of outlier turbines,)
                    y coordinates of (partial) child layout 1.
                y_c2 : array of float64, size: (number of outlier turbines,)
                    y coordinates of (partial) child layout 2.

                """
                
                # find the number of turbines to take from parent 1
                n_wt_1_c1 = np.random.randint(len(x_1))
                n_wt_1_c2 = np.random.randint(len(x_1))
                
                # find the indices of the turbine to take from parent 1
                ind_wt_1_c1 = np.random.randint(len(x_1),size=(n_wt_1_c1))
                ind_wt_1_c2 = np.random.randint(len(x_1),size=(n_wt_1_c2))
                
                # extract turbine child 1
                x_c1 = x_2
                y_c1 = y_2
                x_c1[ind_wt_1_c1] = x_1[ind_wt_1_c1]
                y_c1[ind_wt_1_c1] = y_1[ind_wt_1_c1]
                
                # extract turbine child 2
                x_c2 = x_2
                y_c2 = y_2
                x_c2[ind_wt_1_c2] = x_1[ind_wt_1_c2]
                y_c2[ind_wt_1_c2] = y_1[ind_wt_1_c2]

                return x_c1,x_c2,y_c1,y_c2


            # associate turbines and identify paired turbines and outliers
            
            ind_paired_parent_1,ind_paired_parent_2,ind_outliers_parent_1,ind_outliers_parent_2 = _associate_turbines(x_parent_1,x_parent_2,y_parent_1,y_parent_2)
            
            x_parent_1_paired = x_parent_1[ind_paired_parent_1]
            x_parent_2_paired = x_parent_2[ind_paired_parent_2]
            y_parent_1_paired = y_parent_1[ind_paired_parent_1]
            y_parent_2_paired = y_parent_2[ind_paired_parent_2]
            f_parent_1_paired = f_parent_1[ind_paired_parent_1]
            f_parent_2_paired = f_parent_2[ind_paired_parent_2]
            
            x_parent_1_outliers = x_parent_1[ind_outliers_parent_1]
            x_parent_2_outliers = x_parent_2[ind_outliers_parent_2]
            y_parent_1_outliers = y_parent_1[ind_outliers_parent_1]
            y_parent_2_outliers = y_parent_2[ind_outliers_parent_2]
            
            # linear crossover paired turbines
            if len(x_parent_1_paired)>0:
                x_child_1_paired,x_child_2_paired,y_child_1_paired,y_child_2_paired = _linear_crossover(x_parent_1_paired,x_parent_2_paired,y_parent_1_paired,y_parent_2_paired,f_parent_1_paired,f_parent_2_paired)
            else:
                x_child_1_paired = np.array([])
                x_child_2_paired = np.array([])
                y_child_1_paired = np.array([])
                y_child_2_paired = np.array([])
                
            # random crossover outlier turbines
            if len(x_parent_1_outliers)>0:
                x_child_1_outliers,x_child_2_outliers,y_child_1_outliers,y_child_2_outliers = _random_crossover(x_parent_1_outliers,x_parent_2_outliers,y_parent_1_outliers,y_parent_2_outliers)
            else:
                x_child_1_outliers = np.array([])
                x_child_2_outliers = np.array([])
                y_child_1_outliers = np.array([])
                y_child_2_outliers = np.array([])
                
            # combine child arrays
            x_child_1 = np.concatenate((x_child_1_paired,x_child_1_outliers))
            x_child_2 = np.concatenate((x_child_2_paired,x_child_2_outliers))
            y_child_1 = np.concatenate((y_child_1_paired,y_child_1_outliers))
            y_child_2 = np.concatenate((y_child_2_paired,y_child_2_outliers))
            
            return x_child_1,x_child_2,y_child_1,y_child_2

        
        
        
        # couple parents (randomization inherited from the selection process)
        ind_parent_1 = np.arange(0,self.n_pop,2)
        ind_parent_2 = np.arange(1,self.n_pop,2)
        
        # duplicate the last term in case n_pop is odd (same parents in the last row)
        if self.n_pop%2!=0:
            ind_parent_2 = np.append(ind_parent_2,ind_parent_1[len(ind_parent_1)-1])
                        
        # normalize fitness matrix (to allow negative values for the linear crossover) - normalization along each layout (not along the entire pop)
        fitness_val_mat_parents_min = np.tile(np.min(fitness_val_mat_parents,axis=(1),keepdims=True),(1,self.n_wt))
        fitness_val_mat_parents_max = np.tile(np.max(fitness_val_mat_parents,axis=(1),keepdims=True),(1,self.n_wt))
        fitness_val_mat_parents_norm = np.ones_like(fitness_val_mat_parents)*0.5
        fil_norm = fitness_val_mat_parents_max>fitness_val_mat_parents_min
        fitness_val_mat_parents_norm[fil_norm] = (fitness_val_mat_parents[fil_norm]-fitness_val_mat_parents_min[fil_norm])/(fitness_val_mat_parents_max[fil_norm]-fitness_val_mat_parents_min[fil_norm])
    
        # divide parents matrices
        x_mat_parent_1 = x_mat_parents[ind_parent_1,:]
        x_mat_parent_2 = x_mat_parents[ind_parent_2,:]
        y_mat_parent_1 = y_mat_parents[ind_parent_1,:]
        y_mat_parent_2 = y_mat_parents[ind_parent_2,:]
        fitness_val_mat_parent_1 = fitness_val_mat_parents_norm[ind_parent_1,:]
        fitness_val_mat_parent_2 = fitness_val_mat_parents_norm[ind_parent_2,:]
        
        # apply a crossover function to each couple of parents
        crossover_mat = np.array(list(map(_crossover_function,list(x_mat_parent_1),list(x_mat_parent_2),list(y_mat_parent_1),list(y_mat_parent_2),list(fitness_val_mat_parent_1),list(fitness_val_mat_parent_2))))
        x_mat_child_1 = crossover_mat[:,0,:]
        x_mat_child_2 = crossover_mat[:,1,:]
        y_mat_child_1 = crossover_mat[:,2,:]
        y_mat_child_2 = crossover_mat[:,3,:]
        
        # concatenate children
        x_mat_children = np.concatenate((x_mat_child_1,x_mat_child_2))
        y_mat_children = np.concatenate((y_mat_child_1,y_mat_child_2))
        
        # delete last row in case n_pop is odd
        if self.n_pop%2!=0:
            x_mat_children = x_mat_children[0:x_mat_children.shape[0]-1,:]
            y_mat_children = y_mat_children[0:y_mat_children.shape[0]-1,:]
                    
        return x_mat_children,y_mat_children







    def optimize(self):
        
        """
        This function runs the optimization.
        
        Optimization problem:
            - objective function: fitness_funtion (maximization)
            - variables: (x,y) coordinates
            - constraints: polygonal boundaries (inclusion/exclusion zones)
            
        This version is developed to support a fitness function that receives as input the 
        following variables: (x,y) coordinates        

        Returns
        -------
        x_opt : array of float64, size : (n_wt,)
            optimal x coordinates.
        y_opt : array of float64, size : (n_wt,)
            optimal y coordinates.
        fitness_val_opt : float
            fitness value of the optimal layout defined by (x_opt,y_opt).
        fitness_val_mat_opt : array of float64, size : (n_wt,)
            fitness value of the optimal layout defined by (x_opt,y_opt) for each turbine.
        fitness_val_array_gen : array of float64, size : (n_gen,)
            fitness value of the best layout obtained after each generation.

        """
        
        # initialize output array to save best fitness value of each generation
        fitness_val_array_gen = np.zeros(self.n_gen)

        # define a partial function passing the gen number
        if self.gen_to_f:
            f = partial(self.fitness_function,gen_number_norm=(1)/self.n_gen)
        else:
            f = self.fitness_function
    
        # calculate fitness of the initial population
        if self.full_pop_evaluation:
            fitness_val_mat_initial = f(self.x_mat_initial,self.y_mat_initial)
        else:
            if self.parallel_execution:
                with ProcessPool(self.n_cpu) as pool:
                    results =  pool.map(f,list(self.x_mat_initial),list(self.y_mat_initial))
                fitness_val_mat_initial = np.array(list(results))
            else:
                fitness_val_mat_initial = np.array(list(map(f,list(self.x_mat_initial),list(self.y_mat_initial))))

        # initialize the output
        x_mat_output = self.x_mat_initial
        y_mat_output = self.y_mat_initial
        fitness_val_mat_output = fitness_val_mat_initial

        
        for i in np.arange(0,self.n_gen):
            
            # time
            t_start_iter = time.time()

            # define a partial function passing the gen number
            if self.gen_to_f:
                f = partial(self.fitness_function,gen_number_norm=(i+1)/self.n_gen)
            else:
                f = self.fitness_function

        
            # update values
            x_mat = x_mat_output
            y_mat = y_mat_output
            fitness_val_mat = fitness_val_mat_output
        
            # identify best parents to keep in the next generation
            fitness_val = np.sum(fitness_val_mat,axis=1)
            ind_best_parents = np.argsort(fitness_val)[-self.n_keep_parents:]
            fitness_val_mat_best_parents = fitness_val_mat[ind_best_parents]
            x_mat_best_parents = x_mat[ind_best_parents,:]
            y_mat_best_parents = y_mat[ind_best_parents,:]

            
            # selection
            p_s = self.p_s_array[i]                      # percentage of selection
            x_mat_parents,y_mat_parents,fitness_val_mat_parents = self._selection_layoutOpt(x_mat,y_mat,fitness_val_mat,p_s)
            
            # crossover (+ enforce boundaries)
            x_mat_children_temp,y_mat_children_temp = self._crossover_layoutOpt(x_mat_parents,y_mat_parents,fitness_val_mat_parents)
            x_mat_children,y_mat_children = self.boundaries.enforce_boundaries_MultiPolygon(x_mat_children_temp,y_mat_children_temp)
            
            # calculate fitness value of the children
            if self.full_pop_evaluation:
                fitness_val_mat_children = f(x_mat_children,y_mat_children)
            else:
                if self.parallel_execution:
                    with ProcessPool(self.n_cpu) as pool:
                        results =  pool.map(f,list(x_mat_children),list(y_mat_children))
                    fitness_val_mat_children = np.array(list(results))
                else:
                    fitness_val_mat_children = np.array(list(map(f,list(x_mat_children),list(y_mat_children))))

            # mutation
            p_m = self.p_m_array[i]         # percentage of mutation
            s_m = self.s_m_array[i]         # step limit for mutation
            x_mat_mutated_temp,y_mat_mutated_temp = self._mutation_layoutOpt(x_mat_children,y_mat_children,fitness_val_mat_children,s_m,p_m)
            x_mat_mutated,y_mat_mutated = self.boundaries.enforce_boundaries_MultiPolygon(x_mat_mutated_temp,y_mat_mutated_temp)
            
            # calculate fitness value after mutation
            if self.full_pop_evaluation:
                fitness_val_mat_mutated = f(x_mat_mutated,y_mat_mutated)
            else:
                if self.parallel_execution:
                    with ProcessPool(self.n_cpu) as pool:
                        results =  pool.map(f,list(x_mat_mutated),list(y_mat_mutated))
                    fitness_val_mat_mutated = np.array(list(results))
                else:
                    fitness_val_mat_mutated = np.array(list(map(f,list(x_mat_mutated),list(y_mat_mutated))))
            
            # create output and keep parents (substitute worst children with best parents)
            x_mat_output = x_mat_mutated
            y_mat_output = y_mat_mutated
            fitness_val_mat_output = fitness_val_mat_mutated
            fitness_val_mutated = np.sum(fitness_val_mat_mutated,axis=1)
            if self.n_keep_parents>0:
                ind_worse_children = np.argsort(fitness_val_mutated)[:self.n_keep_parents]
                x_mat_output[ind_worse_children,:] = x_mat_best_parents
                y_mat_output[ind_worse_children,:] = y_mat_best_parents
                fitness_val_mat_output[ind_worse_children,:] = fitness_val_mat_best_parents
            fitness_val_output = np.sum(fitness_val_mat_output,axis=1)
            
            # extract best solution
            ind_opt = np.argmax(fitness_val_output)
            fitness_val_opt = fitness_val_output[ind_opt]
            fitness_val_mat_opt = fitness_val_mat_output[ind_opt,:]
            x_opt = x_mat_output[ind_opt,:]
            y_opt = y_mat_output[ind_opt,:]
            
            # save (and print) result of the iteration
            fitness_val_array_gen[i] = fitness_val_opt
            print(f'Iteration {i} completed - Time required: {round(time.time()-t_start_iter,3)} - Fitness value: {round(fitness_val_opt,5)}')
            
        # save values
        self.x_opt = x_opt
        self.y_opt = y_opt
        self.fitness_val_opt = fitness_val_opt
        self.fitness_val_mat_opt = fitness_val_mat_opt
        self.fitness_val_array_gen = fitness_val_array_gen
        self.x_mat_output = x_mat_output
        self.y_mat_output = y_mat_output

        return x_opt,y_opt,fitness_val_opt,fitness_val_mat_opt,fitness_val_array_gen
    
    
    
    
    
    
#%% MULTI-OBJECTIVE




class LayoutOptimizationGA_MO(LayoutOptimizationGA):
    
    def __init__(
            self,
            fitness_function,
            n_wt,
            boundaries,
            n_obj,
            n_gen,
            n_pop,
            n_keep_parents = 3,
            perc_s = 0.7,
            perc_m = 0.3,
            step_m = 'default',
            distance_limit = 'default',
            x_mat_initial = None,
            y_mat_initial = None,
            parallel_execution = False,
            n_cpu = None,
            full_pop_evaluation = False
            ):

        super().__init__(fitness_function,n_wt,boundaries,n_gen,n_pop,n_keep_parents,perc_s,perc_m,step_m,distance_limit,x_mat_initial,y_mat_initial,parallel_execution,n_cpu,full_pop_evaluation)
        
        self.n_obj = n_obj



    def _extract_Pareto_front(self,fitness_val):
        
        """
        This function extract the Pareto front from a population of solutions and
        calculates the dominance values of each solution.
        The dominance value of a solution is defined as the number of soluntions among
        the population for which this solution is dominant for at least one objective,
        normalized with n_pop-1, so that is in the range [0,1].
        By definition, the Pareto front is the set of solution such that their dominance
        value is equal to 1.
    
        Parameters
        ----------
        fitness_val : array of float64, size : (n_sol,n_obj)
            fitness values for different objectives (n_sol indicates the number of solutions).
    
        Returns
        -------
        ind_pareto : array of float64
            indices of the solutions among the population that correspond to the Pareto front.
        dominance_array : array of float64, size : (n_sol,)
            array containing the dominance value of each solution, i.e. values within the range [0,1].
    
        """
        
        # initialize dominance matrix
        n_sol = fitness_val.shape[0]
        mat_dominance = np.zeros((n_sol,n_sol),dtype='bool')
    
        # iterate for each objective
        for i in np.arange(self.n_obj):
            
            # extract fitness function vector
            f = fitness_val[:,i]
            
            # extend dim of the fitness fucntion vector
            f_ext_1 = np.tile(np.reshape(f,(n_sol,1)),(1,n_sol))
            f_ext_2 = np.tile(np.reshape(f,(1,n_sol)),(n_sol,1))
            
            # compute logical matrix
            mat_dominance = mat_dominance|(f_ext_1>f_ext_2)

        # extract pareto front and calculate dominance array
        dominance_array = np.sum(mat_dominance,axis=1)/(n_sol-1)
        ind_temp = np.arange(0,n_sol)
        ind_pareto = ind_temp[dominance_array==1]
        
        return ind_pareto,dominance_array
    
    
    
    def _calculate_d_crowding(self,fitness_val):
        
        """
        This function calculates the crowding distance for the solutions of a given population
        accorindg to the definition of Martins and Ning, 2022.

        Parameters
        ----------
        fitness_val : array of float64, size : (n_sol,n_obj)
            fitness values for different objectives (n_sol indicates the number of solutions).

        Returns
        -------
        d_crowding : array of float64, size : (n_sol,)
            crowding distance for each solution.

        """
        
        # initialize crowding distance
        n_sol = fitness_val.shape[0]
        d_crowding = np.zeros(n_sol)
        
        # iterate for each objective
        for i in np.arange(self.n_obj):
            
            # initialize value
            d = np.zeros(n_sol)
            
            # extract fitness function vector and sort it
            f = fitness_val[:,i]
            ind_f_sorted = np.argsort(f)
            f_sorted = f[ind_f_sorted]
            
            # check that all the solutions do not have the sane fitness value
            if f_sorted[0]<f_sorted[-1]:
        
                # calculate distance
                f_temp_1 = np.ones(n_sol)*f_sorted[0]
                f_temp_1[1:n_sol] = f_sorted[0:n_sol-1]
                f_temp_2 = np.ones(n_sol)*f_sorted[n_sol-1]
                f_temp_2[0:n_sol-1] = f_sorted[1:n_sol]
                d_sorted = (f_temp_2-f_temp_1)/(f_sorted[n_sol-1]-f_sorted[0])
                d[ind_f_sorted] = d_sorted
            
            # sum the distances
            d_crowding = d_crowding + d
            
        return d_crowding



    def _fitness_values_mat_norm_1(self,f_mat_multiobj):
        
        """
        This function calculates the norm-1 of normalized values (normalization over the entire population).
        It is used to convert the fitness values of multiple objectives into a single information.

        Parameters
        ----------
        f_mat_multiobj : array of float64, size: (n_pop,n_wt,n_obj)
            fitness values.

        Returns
        -------
        f_mat : array of float64, size: (n_pop,n_wt)
            combined fitness values.

        """
        
        # normalize values (along each layout, not along the entire population)
        f_mat_multiobj_min = np.tile(np.min(f_mat_multiobj,axis=(1),keepdims=True),(1,self.n_wt,1))
        f_mat_multiobj_max = np.tile(np.max(f_mat_multiobj,axis=(1),keepdims=True),(1,self.n_wt,1))
        f_mat_multiobj_norm = np.ones_like(f_mat_multiobj)*0.5
        fil = f_mat_multiobj_max>f_mat_multiobj_min
        f_mat_multiobj_norm[fil] = (f_mat_multiobj[fil]-f_mat_multiobj_min[fil])/(f_mat_multiobj_max[fil]-f_mat_multiobj_min[fil])

        # calculate norm 1
        f_mat = np.sum(np.abs(f_mat_multiobj_norm),axis=2)

        return f_mat

        
        
        
        
    def _selection_layoutOpt(self,x_mat,y_mat,fitness_val_mat,p_s):
        
        """
        Multi-ob jective version of the function selection_layoutOpt.
        This function applied a selection to population of layouts based on the "tournament technique".
        It includes a preliminary selection based on the percentage of selection.
        The following criteria are adopted for the selection: (i) dominance value, (ii) crowding distance
        The function is vectorized to compute the operation for all the population.
        The size of the population is identified by n_pop (not required as parameter).

        Parameters
        ----------
        x_mat : array of float64, size : (n_pop,n_wt)
            x coordinates of the turbines.
        y_mat : array of float64, size : (n_pop,n_wt)
            y coordinates of the turbines.
        fitness_val_mat : array of float64, size : (n_pop,n_wt,n_obj)
            value of the fitness function to maximize for each turbine.
        p_s : float
            percentage of selection, expressed as a fraction (i.e. p_s=0.1 --> 10%).

        Returns
        -------
        x_mat_parents : array of float64, size : (n_pop,n_wt)
            x coordinates of the turbines after the selection.
        y_mat_parents : array of float64, size : (n_pop,n_wt)
            y coordinates of the turbines after the selection.
        fitness_val_mat_parents : array of float64, size : (n_pop,n_wt)
            fitness values of the parent population.

        """


        # calculate the fitness of each solution in the population
        fitness_val_array = np.sum(fitness_val_mat,axis=1)

        # calculate dominance array and crowding distance
        ind_pareto,dominance_array = self._extract_Pareto_front(fitness_val_array)
        d_crowding = self._calculate_d_crowding(fitness_val_array)

        # define variable to rank the population conisdering multiple objectives, based on: dominance value (1st criterion) and crowding distance (2nd criterion)
        f_rank_array = dominance_array*(self.n_pop-1)+d_crowding/self.n_obj
        
        # PRELIMINARY STAGE: filter the population based on p_s
        ind_sorted = np.argsort(f_rank_array)
        ind_filtered = ind_sorted[int(np.floor((1-p_s)*self.n_pop)):]

        # complete and shuffle the selected indices
        ind_tournament_A = np.concatenate((ind_filtered,np.random.choice(ind_filtered,size=self.n_pop-len(ind_filtered))))
        ind_tournament_B = np.concatenate((ind_filtered,np.random.choice(ind_filtered,size=self.n_pop-len(ind_filtered))))
        random.shuffle(ind_tournament_A)
        random.shuffle(ind_tournament_B)
        
        # play tournament
        ind_parents = ind_tournament_B
        cond_winner_A = f_rank_array[ind_tournament_A]>f_rank_array[ind_tournament_B]
        ind_parents[cond_winner_A] = ind_tournament_A[cond_winner_A]
        
        # create output
        fitness_val_mat_parents = fitness_val_mat[ind_parents,:,:]
        x_mat_parents = x_mat[ind_parents,:]
        y_mat_parents = y_mat[ind_parents,:]
        
        return x_mat_parents,y_mat_parents,fitness_val_mat_parents







    def _crossover_layoutOpt(self,x_mat_parents,y_mat_parents,fitness_val_mat_parents):
        
        """
        This function iterates the crossover over the entire population.

        Parameters
        ----------
        x_mat_parents : array of float64, size : (n_pop,n_wt)
            x coordinates of the parent layouts.
        y_mat_parents : array of float64, size : (n_pop,n_wt)
            y coordinates of the parent layouts.
        fitness_val_mat_parents : array of float64, size : (n_pop,n_wt,n_obj)
            fitness values of the parent layouts.

        Returns
        -------
        x_mat_children : array of float64, size : (n_pop,n_wt)
            x coordinates of the children layouts.
        y_mat_children : array of float64, size : (n_pop,n_wt)
            y coordinates of the parent layouts.

        """

        def _crossover_function(x_parent_1,x_parent_2,y_parent_1,y_parent_2,f_parent_1,f_parent_2):
            
            """
            This function applies the crossover between two parent solutions (layouts), combining linear and random crossover.
            It can only be applied to two solutions (layouts) individually.
            Working principle: each turbine is identified as paired or outliers, for the 
            former linear crossover is applied while for the latter random crossover is applied.

            Parameters
            ----------
            x_parent_1 : array of float64, size: (n_wt,)
                x coordinates of parent layout 1.
            x_parent_2 : array of float64, size: (n_wt,)
                x coordinates of parent layout 2.
            y_parent_1 : array of float64, size: (n_wt,)
                y coordinates of parent layout 1.
            y_parent_2 : array of float64, size: (n_wt,)
                y coordinates of parent layout 2.
            f_parent_1 : array of float64, size: (n_wt,)
                fitness values of parent layout 1.
            f_parent_2 : array of float64, size: (n_wt,)
                fitness values of parent layout 2.

            Returns
            -------
            x_child_1 : array of float64, size: (n_wt,)
                x coordinates of child layout 1.
            x_child_2 : array of float64, size: (n_wt,)
                x coordinates of child layout 2.
            y_child_1 : array of float64, size: (n_wt,)
                y coordinates of child layout 1.
            y_child_2 : array of float64, size: (n_wt,)
                y coordinates of child layout 2.

            """
            
            def _associate_turbines(x_parent_1,x_parent_2,y_parent_1,y_parent_2):
                
                """
                This function identifies which turbines occupy the same location (within a radius equal to d_limit) between two different layouts.
                It can only be applied to two solutions (layouts) individually.
                Working principle: for each turbine of the layout 1, the closest turbine of layout 2 is associated if its distance is lower than d_limit.
                Limitation: if two turbines of layout 1 have the same closest turbine, the first turbine of layout 1 is prioritized and the second turbine is appointed as outlier.

                Parameters
                ----------
                x_parent_1 : array of float64, size : (n_wt,)
                    x coordinates of layout 1.
                x_parent_2 : array of float64, size : (n_wt,)
                    y coordinates of layout 1.
                y_parent_1 : array of float64, size : (n_wt,)
                    x coordinates of layout 2.
                y_parent_2 : array of float64, size : (n_wt,)
                    y coordinates of layout 2.

                Returns
                -------
                ind_paired_parent_1 : array of int, size: (number of paired turbines,)
                    indices of the paired turbines of layout 1 with layout 2 (ordered).
                ind_paired_parent_2 : array of int, size: (number of paired turbines,)
                    indices of the paired turbines of layout 2 with layout 1 (ordered).
                ind_outliers_parent_1 : array of int, size: (number of outlier turbines,)
                    indices of the outlier turbines of layout 1.
                ind_outliers_parent_2 : array of int, size: (number of outlier turbines,)
                    indices of the outlier turbines of layout 2.

                """

                n_wt = len(x_parent_1)
                
                # calculate distance
                x_parent_1_ext = np.tile(np.reshape(x_parent_1,(1,n_wt)),(n_wt,1))
                x_parent_2_ext = np.tile(np.reshape(x_parent_2,(n_wt,1)),(1,n_wt))
                y_parent_1_ext = np.tile(np.reshape(y_parent_1,(1,n_wt)),(n_wt,1))
                y_parent_2_ext = np.tile(np.reshape(y_parent_2,(n_wt,1)),(1,n_wt))
                d_mat = np.sqrt((x_parent_1_ext-x_parent_2_ext)**2+(y_parent_1_ext-y_parent_2_ext)**2)
                
                # create mask to identify only the nearest turbine of each turbine of parent 1
                ind_d_min_parent_1 = np.argmin(d_mat,axis=0)
                ind_mat_temp = np.tile(np.reshape(np.arange(0,n_wt),(n_wt,1)),(1,n_wt))
                ind_d_min_parent_1_ext = np.tile(np.reshape(ind_d_min_parent_1,(1,n_wt)),(n_wt,1))
                ind_d_min_parent_1_ext[ind_d_min_parent_1_ext!=ind_mat_temp] = -1
                d_mat_filtered_temp = np.inf*np.ones((n_wt,n_wt))
                d_mat_filtered_temp[ind_d_min_parent_1_ext>=0] = d_mat[ind_d_min_parent_1_ext>=0]
                
                # avoid duplicates
                d_mat_filtered = np.inf*np.ones((n_wt,n_wt))
                d_mat_filtered[np.arange(0,n_wt),np.argmin(d_mat_filtered_temp,axis=1)] = d_mat_filtered_temp[np.arange(0,n_wt),np.argmin(d_mat_filtered_temp,axis=1)]
                
                # obtain the index of the nearest turbine for each turbine of parent 1 avoiding duplicates (-1 indicates outliers)
                ind_d_min_parent_1_filtered = np.argmin(d_mat_filtered,axis=0)
                ind_d_min_parent_1_filtered[np.min(d_mat_filtered,axis=0)>self.d_limit] = -1
                
                # find paired values
                ind_paired_parent_1 = np.arange(0,n_wt)[ind_d_min_parent_1_filtered>=0]
                ind_paired_parent_2 = ind_d_min_parent_1_filtered[ind_d_min_parent_1_filtered>=0]
                
                # find outliers parent 1
                ind_outliers_parent_1 = np.arange(0,n_wt)[ind_d_min_parent_1_filtered<0]
                
                # find outliers parent 2
                ind_mat_temp = np.tile(np.reshape(np.arange(0,n_wt),(n_wt,1)),(1,n_wt))
                ind_d_min_parent_1_filtered_ext = np.tile(np.reshape(ind_d_min_parent_1_filtered,(1,n_wt)),(n_wt,1))
                ind_mat_temp_filtered = -np.ones((n_wt,n_wt),dtype=int)
                ind_mat_temp_filtered[ind_mat_temp==ind_d_min_parent_1_filtered_ext] = ind_d_min_parent_1_filtered_ext[ind_mat_temp==ind_d_min_parent_1_filtered_ext]
                ind_paired_parent_2_ordered_temp = np.max(ind_mat_temp_filtered,axis=1) 
                ind_outliers_parent_2 = np.arange(0,n_wt)[ind_paired_parent_2_ordered_temp<0]
                
                return ind_paired_parent_1,ind_paired_parent_2,ind_outliers_parent_1,ind_outliers_parent_2


            def _linear_crossover(x_1,x_2,y_1,y_2,f_1,f_2):
                
                """
                This function applies the linear crossover between two parent (partial) layouts.
                It can only be applied to two solutions (layouts) individually.
                Working prinicple: both children are obtained along the line that connects the two paired turbines, the first one is in between (depending on f_1,f_2) and the second is extrpolated towards the best turbine
                
                Parameters
                ----------
                x_1 : array of float64, size: (number of paired turbines,)
                    x coordinates of (partial) parent layout 1.
                x_2 : array of float64, size: (number of paired turbines,)
                    x coordinates of (partial) parent layout 2.
                y_1 : array of float64, size: (number of paired turbines,)
                    y coordinates of (partial) parent layout 1.
                y_2 : array of float64, size: (number of paired turbines,)
                    y coordinates of (partial) parent layout 2.
                f_1 : array of float64, size: (number of paired turbines,)
                    fitness function values correspondent to the turbines identified by (x_1,y_1).
                f_2 : array of float64, size: (number of paired turbines,)
                    fitness function values correspondent to the turbines identified by (x_2,y_2).

                Returns
                -------
                x_c1 : array of float64, size: (number of paired turbines,)
                    x coordinates of (partial) child layout 1.
                x_c2 : array of float64, size: (number of paired turbines,)
                    x coordinates of (partial) child layout 2.
                y_c1 : array of float64, size: (number of paired turbines,)
                    y coordinates of (partial) child layout 1.
                y_c2 : array of float64, size: (number of paired turbines,)
                    y coordinates of (partial) child layout 2.
                    
                """
                
                # avoid condition f_1=f_2=0
                check = (f_1==0) & (f_2==0)
                f_1[check] = 1
                f_2[check] = 1
                
                # point between the turbines
                x_c1 = (f_1/(f_1+f_2))*x_1+(f_2/(f_1+f_2))*x_2
                y_c1 = (f_1/(f_1+f_2))*y_1+(f_2/(f_1+f_2))*y_2
                
                # point on the side of the better turbine
                x_c2_temp_1 = x_1+(x_1-x_2)*(f_1/(f_1+f_2))
                y_c2_temp_1 = y_1+(y_1-y_2)*(f_1/(f_1+f_2))
                x_c2_temp_2 = x_2+(x_2-x_1)*(f_2/(f_1+f_2))
                y_c2_temp_2 = y_2+(y_2-y_1)*(f_2/(f_1+f_2))
                x_c2 = x_c2_temp_1
                x_c2[f_2>f_1] = x_c2_temp_2[f_2>f_1]
                y_c2 = y_c2_temp_1
                y_c2[f_2>f_1] = y_c2_temp_2[f_2>f_1]
                
                return x_c1,x_c2,y_c1,y_c2


            def _random_crossover(x_1,x_2,y_1,y_2):
                
                """
                This function applies the random crossover between two parent (partial) layouts.
                It can only be applied to two solutions (layouts) individually.
                Working principle: the turbine of the children are selected randomly between the parent layouts.

                Parameters
                ----------
                x_1 : array of float64, size: (number of outlier turbines,)
                    x coordinates of (partial) parent layout 1.
                x_2 : array of float64, size: (number of outlier turbines,)
                    x coordinates of (partial) parent layout 2.
                y_1 : array of float64, size: (number of outlier turbines,)
                    y coordinates of (partial) parent layout 1.
                y_2 : array of float64, size: (number of outlier turbines,)
                    y coordinates of (partial) parent layout 2.

                Returns
                -------
                x_c1 : array of float64, size: (number of outlier turbines,)
                    x coordinates of (partial) child layout 1.
                x_c2 : array of float64, size: (number of outlier turbines,)
                    x coordinates of (partial) child layout 2.
                y_c1 : array of float64, size: (number of outlier turbines,)
                    y coordinates of (partial) child layout 1.
                y_c2 : array of float64, size: (number of outlier turbines,)
                    y coordinates of (partial) child layout 2.

                """
                
                # find the number of turbines to take from parent 1
                n_wt_1_c1 = np.random.randint(len(x_1))
                n_wt_1_c2 = np.random.randint(len(x_1))
                
                # find the indices of the turbine to take from parent 1
                ind_wt_1_c1 = np.random.randint(len(x_1),size=(n_wt_1_c1))
                ind_wt_1_c2 = np.random.randint(len(x_1),size=(n_wt_1_c2))
                
                # extract turbine child 1
                x_c1 = x_2
                y_c1 = y_2
                x_c1[ind_wt_1_c1] = x_1[ind_wt_1_c1]
                y_c1[ind_wt_1_c1] = y_1[ind_wt_1_c1]
                
                # extract turbine child 2
                x_c2 = x_2
                y_c2 = y_2
                x_c2[ind_wt_1_c2] = x_1[ind_wt_1_c2]
                y_c2[ind_wt_1_c2] = y_1[ind_wt_1_c2]

                return x_c1,x_c2,y_c1,y_c2


            # associate turbines and identify paired turbines and outliers
            
            ind_paired_parent_1,ind_paired_parent_2,ind_outliers_parent_1,ind_outliers_parent_2 = _associate_turbines(x_parent_1,x_parent_2,y_parent_1,y_parent_2)
            
            x_parent_1_paired = x_parent_1[ind_paired_parent_1]
            x_parent_2_paired = x_parent_2[ind_paired_parent_2]
            y_parent_1_paired = y_parent_1[ind_paired_parent_1]
            y_parent_2_paired = y_parent_2[ind_paired_parent_2]
            
            f_parent_1_paired = f_parent_1[ind_paired_parent_1]
            f_parent_2_paired = f_parent_2[ind_paired_parent_2]
            
            x_parent_1_outliers = x_parent_1[ind_outliers_parent_1]
            x_parent_2_outliers = x_parent_2[ind_outliers_parent_2]
            y_parent_1_outliers = y_parent_1[ind_outliers_parent_1]
            y_parent_2_outliers = y_parent_2[ind_outliers_parent_2]
                        
            # linear crossover paired turbines
            if len(x_parent_1_paired)>0:
                x_child_1_paired,x_child_2_paired,y_child_1_paired,y_child_2_paired = _linear_crossover(x_parent_1_paired,x_parent_2_paired,y_parent_1_paired,y_parent_2_paired,f_parent_1_paired,f_parent_2_paired)
            else:
                x_child_1_paired = np.array([])
                x_child_2_paired = np.array([])
                y_child_1_paired = np.array([])
                y_child_2_paired = np.array([])
                
            # random crossover outlier turbines
            if len(x_parent_1_outliers)>0:
                x_child_1_outliers,x_child_2_outliers,y_child_1_outliers,y_child_2_outliers = _random_crossover(x_parent_1_outliers,x_parent_2_outliers,y_parent_1_outliers,y_parent_2_outliers)
            else:
                x_child_1_outliers = np.array([])
                x_child_2_outliers = np.array([])
                y_child_1_outliers = np.array([])
                y_child_2_outliers = np.array([])
                
            # combine child arrays
            x_child_1 = np.concatenate((x_child_1_paired,x_child_1_outliers))
            x_child_2 = np.concatenate((x_child_2_paired,x_child_2_outliers))
            y_child_1 = np.concatenate((y_child_1_paired,y_child_1_outliers))
            y_child_2 = np.concatenate((y_child_2_paired,y_child_2_outliers))
            
            return x_child_1,x_child_2,y_child_1,y_child_2

        
        
        
        # couple parents (randomization inherited from the selection process)
        ind_parent_1 = np.arange(0,self.n_pop,2)
        ind_parent_2 = np.arange(1,self.n_pop,2)
        
        # duplicate the last term in case n_pop is odd (same parents in the last row)
        if self.n_pop%2!=0:
            ind_parent_2 = np.append(ind_parent_2,ind_parent_1[len(ind_parent_1)-1])
        
        # normalize fitness matrix and calculate norm 1 (to allow negative values and convert into one objective information)
        fitness_val_mat_parents_norm = self._fitness_values_mat_norm_1(fitness_val_mat_parents)
        
        # divide parents matrices
        x_mat_parent_1 = x_mat_parents[ind_parent_1,:]
        x_mat_parent_2 = x_mat_parents[ind_parent_2,:]
        y_mat_parent_1 = y_mat_parents[ind_parent_1,:]
        y_mat_parent_2 = y_mat_parents[ind_parent_2,:]
        fitness_val_mat_parent_1 = fitness_val_mat_parents_norm[ind_parent_1,:]
        fitness_val_mat_parent_2 = fitness_val_mat_parents_norm[ind_parent_2,:]
        
        # apply a crossover function to each couple of parents
        crossover_mat = np.array(list(map(_crossover_function,list(x_mat_parent_1),list(x_mat_parent_2),list(y_mat_parent_1),list(y_mat_parent_2),list(fitness_val_mat_parent_1),list(fitness_val_mat_parent_2))))
        x_mat_child_1 = crossover_mat[:,0,:]
        x_mat_child_2 = crossover_mat[:,1,:]
        y_mat_child_1 = crossover_mat[:,2,:]
        y_mat_child_2 = crossover_mat[:,3,:]
        
        # concatenate children
        x_mat_children = np.concatenate((x_mat_child_1,x_mat_child_2))
        y_mat_children = np.concatenate((y_mat_child_1,y_mat_child_2))
        
        # delete last row in case n_pop is odd
        if self.n_pop%2!=0:
            x_mat_children = x_mat_children[0:x_mat_children.shape[0]-1,:]
            y_mat_children = y_mat_children[0:y_mat_children.shape[0]-1,:]
                    
        return x_mat_children,y_mat_children





    def _mutation_layoutOpt(self,x_mat,y_mat,fitness_val_mat,s_m,p_m):
        
        """
        Multi-objective version of the function mutation_layoutOpt.
        This function applies a mutation to a layout identified by x,y arrays.
        The function is vectorized to compute the operation for all the population.
        The size of the population is identified by n_pop (not required as parameter).
    
        Parameters
        ----------
        x_mat : array of float64, size : (n_pop,n_wt)
            x coordinates of the turbines.
        y_mat : array of float64, size : (n_pop,n_wt)
            y coordinates of the turbines.
        fitness_val_mat_1 : array of float64, size : (n_pop,n_wt)
            value of the fitness function (objective 1) to maximize for each turbine.
        fitness_val_mat_2 : array of float64, size : (n_pop,n_wt)
            value of the fitness function (objective 2) to maximize for each turbine.
        s_m : float
            max value for the random step of the mutation.
        p_m : float
            percentage of mutation for an individual turbine (e.g. p_m=0.1 --> 1 turbine out of 10 is relocated), expressed as a fraction (i.e. p_m=0.1 --> 10%).
    
        Returns
        -------
        x_mat_new : array of float64, size : (n_pop,n_wt)
            x coordinates of the turbines after the mutation.
        y_mat_new : array of float64, size : (n_pop,n_wt)
            y coordinates of the turbines after the mutation.
    
        """
        
        
        # convert multi-obj. fitness values into a single fitness value
        fitness_val_mat_compressed = self._fitness_values_mat_norm_1(fitness_val_mat)

        # rank the turbines from higher to lower fitness
        ind_ranked = np.argsort(-fitness_val_mat_compressed,axis=1)
        
        # create probability matrix (LINEAR INCREASE)
        p_val_array = np.minimum(self.n_wt*p_m*np.linspace(0,1,self.n_wt)/np.sum(np.linspace(0,1,self.n_wt)),1)
        p_val_mat = np.tile(np.reshape(p_val_array,(1,len(p_val_array))),(self.n_pop,1))
        
        # assign probability values to each turbine (high probability for low fitness value -> these turbines will mutate)
        ind_col = ind_ranked.reshape(-1)
        ind_row = np.tile(np.reshape(np.arange(0,self.n_pop),(self.n_pop,1)),(1,self.n_wt)).reshape(-1)
        p_mat_temp = np.zeros((self.n_pop,self.n_wt))
        p_mat_temp[ind_row,ind_col] = p_val_mat.reshape(-1)
        p_mat = np.reshape(p_mat_temp,(self.n_pop,self.n_wt))
        
        # create mutation bool matrix (0 = no mutation, 1 = mutation)
        mutation_bool_mat = np.random.rand(self.n_pop,self.n_wt)<p_mat
        mutation_int_mat = np.zeros((self.n_pop,self.n_wt))
        mutation_int_mat[mutation_bool_mat] = 1
        
        # generate random step and direction
        step = s_m*np.random.rand(self.n_pop,self.n_wt)
        direction = ((2*np.pi))*np.random.rand(self.n_pop,self.n_wt)
        
        # apply mutation
        x_mat_new = x_mat+mutation_int_mat*step*np.cos(direction)
        y_mat_new = y_mat+mutation_int_mat*step*np.sin(direction)
        
        return x_mat_new,y_mat_new
    






    def optimize(self):

        """
        This function runs the optimization.
        
        Optimization problem:
            - objective function: multiple fitness_funtion (maximization)
            - variables: (x,y) coordinates
            - constraints: polygonal boundaries (inclusion/exclusion zones)
            
        This version is developed to support a fitness function that receives as input the 
        following variables: (x,y) coordinates        

        Returns
        -------
        x_mat_output_pareto : array of float64, size : (size_pareto,n_wt)
            optimal x coordinates of the solutions in the Pareto front.
        y_mat_output_pareto : array of float64, size : (size_pareto,n_wt)
            optimal y coordinates of the solutions in the Pareto front.
        fitness_val_output_pareto : array of float64, size : (size_pareto,n_obj)
            fitness value of the solutions in the Pareto front.

        """
            
        # calculate fitness of the initial population (iterate for each objective)
        if self.full_pop_evaluation:
            fitness_val_mat_initial = self.fitness_function(self.x_mat_initial,self.y_mat_initial)
        else:
            if self.parallel_execution:
                with ProcessPool(self.n_cpu) as pool:
                    results =  pool.map(self.fitness_function,list(self.x_mat_initial),list(self.y_mat_initial))
                fitness_val_mat_initial = np.array(list(results)).transpose(0,2,1)
            else:
                results = map(self.fitness_function,list(self.x_mat_initial),list(self.y_mat_initial))
                fitness_val_mat_initial = np.array(list(results)).transpose(0,2,1)

            
        # calculate the fitness of each solution and extract Pareto front
        fitness_val_initial = np.sum(fitness_val_mat_initial,axis=1)
        ind_pareto,dominance_vec = self._extract_Pareto_front(fitness_val_initial)
        
        # initialize the output
        x_mat_output = self.x_mat_initial
        y_mat_output = self.y_mat_initial
        fitness_val_mat_output = fitness_val_mat_initial
        
        # intialize the Pareto front 
        fitness_val_mat_output_pareto = fitness_val_mat_output[ind_pareto,:,:]
        fitness_val_output_pareto = np.sum(fitness_val_mat_output_pareto,axis=1)
        x_mat_output_pareto = x_mat_output[ind_pareto,:]
        y_mat_output_pareto = y_mat_output[ind_pareto,:]

        # iterate for each generation
        for i in np.arange(0,self.n_gen):
            
            # time
            t_start_iter = time.time()
        
            # update values
            x_mat = x_mat_output
            y_mat = y_mat_output
            fitness_val_mat = fitness_val_mat_output
            fitness_val_mat_pareto = fitness_val_mat_output_pareto
            fitness_val_pareto = fitness_val_output_pareto
            x_mat_pareto = x_mat_output_pareto
            y_mat_pareto = y_mat_output_pareto

            # selection
            p_s = self.p_s_array[i]                      # percentage of selection
            x_mat_parents,y_mat_parents,fitness_val_mat_parents = self._selection_layoutOpt(x_mat,y_mat,fitness_val_mat,p_s)
            
            # crossover (+ enforce boundaries)
            x_mat_children_temp,y_mat_children_temp = self._crossover_layoutOpt(x_mat_parents,y_mat_parents,fitness_val_mat_parents)
            x_mat_children,y_mat_children = self.boundaries.enforce_boundaries_MultiPolygon(x_mat_children_temp,y_mat_children_temp)
            
            # calculate fitness of the children population (iterate for each objective)
            if self.full_pop_evaluation:
                fitness_val_mat_children = self.fitness_function(x_mat_children,y_mat_children)
            else:
                if self.parallel_execution:
                    with ProcessPool(self.n_cpu) as pool:
                        results =  pool.map(self.fitness_function,list(x_mat_children),list(y_mat_children))
                    fitness_val_mat_children = np.array(list(results)).transpose(0,2,1)
                else:
                    results = map(self.fitness_function,list(x_mat_children),list(y_mat_children))
                    fitness_val_mat_children = np.array(list(results)).transpose(0,2,1)

            # mutation
            p_m = self.p_m_array[i]     # percentage of mutation
            s_m = self.s_m_array[i]     # step limit for mutation
            x_mat_mutated_temp,y_mat_mutated_temp = self._mutation_layoutOpt(x_mat_children,y_mat_children,fitness_val_mat_children,s_m,p_m)
            x_mat_mutated,y_mat_mutated = self.boundaries.enforce_boundaries_MultiPolygon(x_mat_mutated_temp,y_mat_mutated_temp)

            # calculate fitness after mutation (iterate for each objective)
            if self.full_pop_evaluation:
                fitness_val_mat_mutated = self.fitness_function(x_mat_mutated,y_mat_mutated)
            else:
                if self.parallel_execution:
                    with ProcessPool(self.n_cpu) as pool:
                        results =  pool.map(self.fitness_function,list(x_mat_mutated),list(y_mat_mutated))
                    fitness_val_mat_mutated = np.array(list(results)).transpose(0,2,1)
                else:
                    results = map(self.fitness_function,list(x_mat_mutated),list(y_mat_mutated))
                    fitness_val_mat_mutated = np.array(list(results)).transpose(0,2,1)
            fitness_val_mutated = np.sum(fitness_val_mat_mutated,axis=1)

            # create extended output (concatenate with previous Pareto front)
            fitness_val_output_ext = np.concatenate((fitness_val_mutated,fitness_val_pareto),axis=0)
            fitness_val_mat_output_ext = np.concatenate((fitness_val_mat_mutated,fitness_val_mat_pareto),axis=0)
            x_mat_output_ext = np.concatenate((x_mat_mutated,x_mat_pareto),axis=0)
            y_mat_output_ext = np.concatenate((y_mat_mutated,y_mat_pareto),axis=0)
            
            # create output (filter best solutions to keep n_pop size) 
            _,dominance_array_output_ext = self._extract_Pareto_front(fitness_val_output_ext)
            d_crowding_output_ext = self._calculate_d_crowding(fitness_val_output_ext)
            f_rank_array = dominance_array_output_ext*(self.n_pop-1)+d_crowding_output_ext/self.n_obj
            ind_sorted = np.argsort(-f_rank_array)
            ind_filtered = ind_sorted[0:self.n_pop]
            x_mat_output = x_mat_output_ext[ind_filtered,:]
            y_mat_output = y_mat_output_ext[ind_filtered,:]
            fitness_val_mat_output = fitness_val_mat_output_ext[ind_filtered,:,:]
            fitness_val_output = np.sum(fitness_val_mat_output,axis=1)
            
            # save Pareto front 
            ind_pareto,_ = self._extract_Pareto_front(fitness_val_output)
            fitness_val_output_pareto = fitness_val_output[ind_pareto,:]
            fitness_val_mat_output_pareto = fitness_val_mat_output[ind_pareto,:,:]
            x_mat_output_pareto = x_mat_output[ind_pareto,:]
            y_mat_output_pareto = y_mat_output[ind_pareto,:]
                    
            print(f'Iteration {i} completed - Time required: {round(time.time()-t_start_iter,3)} - Size of Pareto front: {len(ind_pareto)}')
            
        # save values
        self.x_opt = x_mat_output_pareto
        self.y_opt = y_mat_output_pareto
        self.fitness_val_opt = fitness_val_output_pareto

        return x_mat_output_pareto,y_mat_output_pareto,fitness_val_output_pareto

        
        
        


#%%


# # import py_wake packages
# from py_wake.examples.data.dtu10mw import DTU10MW
# from py_wake.wind_farm_models import PropagateDownwind
# from py_wake.site import UniformWeibullSite
# from py_wake.deficit_models import BastankhahGaussianDeficit
# from py_wake.superposition_models import SquaredSum
# from py_wake.deflection_models import JimenezWakeDeflection
# from py_wake.rotor_avg_models import GaussianOverlapAvgModel

# import matplotlib.pyplot as plt



# if __name__ == '__main__':

#     # deifne site (HKN)
#     wd_site = np.linspace(0,360,12,endpoint=False)
#     p_wd_site = np.array([0.066,0.063,0.063,0.064,0.054,0.052,0.072,0.129,0.150,0.116,0.091,0.080])
#     a_site = np.array([9.56,9.21,9.38,9.78,9.23,9.20,10.96,12.73,12.75,12.17,11.22,10.59])
#     k_site = np.array([2.18,2.36,2.40,2.34,2.30,2.20,2.11,2.33,2.42,2.20,2.15,2.11])
#     site = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.1)
    
#     # define wind turbine
#     wind_turbine = DTU10MW()
#     diameter = wind_turbine.diameter()
#     p_rated = 10
#     ws_rated = 11.4
    
#     # define wind farm model
#     wfm = PropagateDownwind(site, wind_turbine,
#                             wake_deficitModel=BastankhahGaussianDeficit(),
#                             superpositionModel=SquaredSum(),
#                             deflectionModel=JimenezWakeDeflection(),
#                             turbulenceModel=None,
#                             rotorAvgModel=GaussianOverlapAvgModel())
    
#     # define case study
#     min_d = 0.5*diameter
#     wd_array = np.arange(0,360,5)
#     ws_array = np.arange(3,26)
    
    
#     # define case study
#     n_wt = 9
#     p_density = 20
    
    
#     # optimization parameters (general)
#     n_pop = 48                         # size of the population
#     n_gen = 5                         # number of generations
#     n_keep_parents = 3                  # number of parents kept from the previous generation    
    
    
#     # optimization parameters
#     p_s_min = 0.7                     # min percentage of selection
#     p_s_max = 0.7                     # max percentage of selection
#     p_m_min = 0.1                    # min percentage of mutation
#     p_m_max = 0.3                     # max percentage of mutation
#     s_m_min = 0*diameter            # min step of mutation
#     s_m_max = 3*diameter              # max step of mutation
#     d_limit = 1*diameter              # distance limit for turbine association during crossover
    
    
#     # create boundaries
    
#     x_boundaries_polygon_1 = np.array([0,200,1500,2700,1600,1650,0],dtype=float)
#     y_boundaries_polygon_1 = np.array([40,1800,2500,2300,750,-100,40],dtype=float)
    
#     x_boundaries_polygon_2 = np.array([2000,2100,2400,2900,2000],dtype=float)
#     y_boundaries_polygon_2 = np.array([-100,500,1000,20,-100],dtype=float)
    
#     x_exclusion_zone_1 = np.array([750,830,1250,1340,900,750])
#     y_exclusion_zone_1 = np.array([750,1300,1250,840,1000,750])
    
#     x_exclusion_zone_2 = np.array([1600,2100,2350,1600])
#     y_exclusion_zone_2 = np.array([1600,2350,2050,1600])
    
#     x_boundaries_polygon_list = [x_boundaries_polygon_1,x_boundaries_polygon_2,x_exclusion_zone_1,x_exclusion_zone_2]
#     y_boundaries_polygon_list = [y_boundaries_polygon_1,y_boundaries_polygon_2,y_exclusion_zone_1,y_exclusion_zone_2]
#     convex_list = [False,True,False,False]
#     exclusion_list = [False,False,True,True]
    
#     boundaries = Boundaries(x_boundaries_polygon_list,y_boundaries_polygon_list,convex_list,exclusion_list)


#     # create a bathymetry grid (EXAMPLE)
#     x_bathymetry = np.linspace(-500,3500,100)
#     y_bathymetry = np.linspace(-500,3500,100)
#     x_grid,y_grid = np.meshgrid(x_bathymetry,y_bathymetry,indexing='ij')
#     def irregular_bathymetry(x,y):
#         return ((np.sin(x/1000)*np.cos(y/1000))+(0.5*np.sin((x*y)/1000000))-2)*20
#     water_depth_bathymetry = irregular_bathymetry(x_grid,y_grid)




#     # define fitness function wrapper: AEP

#     class AEPFunctionWrapper():

#         def __init__(self,min_d,wfm,wd_array,ws_array,diameter,ws_rated):
#             self.min_d = min_d
#             self.wfm = wfm
#             self.wd_array = wd_array
#             self.ws_array = ws_array
#             self.diameter = diameter
#             self.ws_rated = ws_rated

#         def filter_turbines(self,x,y):
        
#             # create distance matrix
#             x_mat_1 = np.tile(np.reshape(x,(1,len(x))),(len(x),1))
#             x_mat_2 = np.tile(np.reshape(x,(len(x),1)),(1,len(x)))
#             y_mat_1 = np.tile(np.reshape(y,(1,len(y))),(len(y),1))
#             y_mat_2 = np.tile(np.reshape(y,(len(y),1)),(1,len(y)))
#             d_mat = np.sqrt((x_mat_1-x_mat_2)**2+(y_mat_1-y_mat_2)**2)
        
#             # identify the turbines that have d<d_lim
#             ind_mat = np.tile(np.reshape(np.arange(0,len(x)),(1,len(x))),(len(x),1))
#             ind_mat[d_mat>=self.min_d] = -1
#             ind_mat[np.triu(np.ones((len(x),len(x))),1)<1] = -1
#             ind_turbine_delete_temp = ind_mat[ind_mat>=0]
        
#             # create x,y new vectors with only relevant turbines
#             ind_turbine_delete = np.unique(ind_turbine_delete_temp)
#             ind_keep = np.delete(np.arange(0,len(x)),ind_turbine_delete)
#             x_new = x[ind_keep]
#             y_new = y[ind_keep]
        
#             return x_new,y_new,ind_keep
    
#         def calculate_AEP(self,x,y):
#             x_new,y_new,ind_keep = self.filter_turbines(x,y)
#             aep_temp = np.sum(self.wfm(x_new,y_new,wd=self.wd_array,ws=self.ws_array,yaw=0,tilt=0).aep_ilk(),axis=(1, 2)) 
#             aep = np.zeros(len(x))
#             aep[ind_keep] = aep_temp    
#             return aep
        
#         def __call__(self,x,y):
#             return self.calculate_AEP(x,y)
        

#     # create instance for AEP fitness function wrapper
#     aep_wrapper = AEPFunctionWrapper(min_d,wfm,wd_array,ws_array,diameter,ws_rated)





#     # define fitness function wrapper: COE_CO2

#     class CO2FunctionWrapper():

#         def __init__(self,min_d,wfm,wd_array,ws_array,diameter,ws_rated,x_bathymetry,y_bathymetry,water_depth_bathymetry):
#             self.min_d = min_d
#             self.wfm = wfm
#             self.wd_array = wd_array
#             self.ws_array = ws_array
#             self.diameter = diameter
#             self.ws_rated = ws_rated
#             self.x_bathymetry = x_bathymetry
#             self.y_bathymetry = y_bathymetry
#             self.water_depth_bathymetry = water_depth_bathymetry

#         def filter_turbines(self,x,y):
        
#             # create distance matrix
#             x_mat_1 = np.tile(np.reshape(x,(1,len(x))),(len(x),1))
#             x_mat_2 = np.tile(np.reshape(x,(len(x),1)),(1,len(x)))
#             y_mat_1 = np.tile(np.reshape(y,(1,len(y))),(len(y),1))
#             y_mat_2 = np.tile(np.reshape(y,(len(y),1)),(1,len(y)))
#             d_mat = np.sqrt((x_mat_1-x_mat_2)**2+(y_mat_1-y_mat_2)**2)
        
#             # identify the turbines that have d<d_lim
#             ind_mat = np.tile(np.reshape(np.arange(0,len(x)),(1,len(x))),(len(x),1))
#             ind_mat[d_mat>=self.min_d] = -1
#             ind_mat[np.triu(np.ones((len(x),len(x))),1)<1] = -1
#             ind_turbine_delete_temp = ind_mat[ind_mat>=0]
        
#             # create x,y new vectors with only relevant turbines
#             ind_turbine_delete = np.unique(ind_turbine_delete_temp)
#             ind_keep = np.delete(np.arange(0,len(x)),ind_turbine_delete)
#             x_new = x[ind_keep]
#             y_new = y[ind_keep]
        
#             return x_new,y_new,ind_keep
    
#         def calculate_AEP(self,x,y):
#             x_new,y_new,ind_keep = self.filter_turbines(x,y)
#             aep_temp = np.sum(self.wfm(x_new,y_new,wd=self.wd_array,ws=self.ws_array,yaw=0,tilt=0).aep_ilk(),axis=(1, 2)) 
#             aep = np.zeros(len(x))
#             aep[ind_keep] = aep_temp    
#             return aep

#         def calculate_water_depth(self,x,y):
#             from scipy.interpolate import RegularGridInterpolator
#             interp_function = RegularGridInterpolator((self.x_bathymetry,self.y_bathymetry),self.water_depth_bathymetry)
#             water_depth = interp_function((x,y))
#             return water_depth

#         def cost_of_energy_co2(self,x,y):
#             water_depth = self.calculate_water_depth(x,y)
#             emissions = 3000+(water_depth**2)*0.5
#             aep = self.calculate_AEP(x,y)
#             return emissions/np.maximum(aep,1e-5)
        
#         def __call__(self,x,y):
#             return -self.cost_of_energy_co2(x,y)


#     # create instance for CO2 fitness function wrapper
#     co2_wrapper = CO2FunctionWrapper(min_d,wfm,wd_array,ws_array,diameter,ws_rated,x_bathymetry,y_bathymetry,water_depth_bathymetry)


#     # combine objectives for 
#     f_wrapper_MO = [aep_wrapper,co2_wrapper]




#     # create optimization object
#     layout_optimization = LayoutOptimizationGA_MO(f_wrapper_MO,
#                                             n_wt,
#                                             boundaries,
#                                             n_obj=2,
#                                             n_gen=n_gen,
#                                             n_pop=n_pop,
#                                             perc_s=0.7,
#                                             perc_m=np.array([p_m_max,p_m_min]),
#                                             step_m=np.array([s_m_max,s_m_min]),
#                                             distance_limit=d_limit,
#                                             parallel_execution=True,
#                                             n_cpu=12
#                                             )
        


#     # run optmization
#     t_1 = time.time()
#     x_opt,y_opt,_ = layout_optimization.optimize()
#     t_2 = time.time()
#     f_Pareto = layout_optimization.fitness_val_opt
    
#     print(f_Pareto)
#     print(t_2-t_1)











    
#     ## create optimization object
#     #layout_optimization = LayoutOptimizationGA(aep_wrapper,#fitness_function,
#     #                                           n_wt,
#     #                                           boundaries,
#     #                                           n_gen,
#     #                                           n_pop,
#     #                                           perc_s=0.7,
#     #                                           perc_m=np.array([p_m_max,p_m_min]),
#     #                                           step_m=np.array([s_m_max,s_m_min]),
#     #                                           distance_limit=d_limit,
#     #                                           parallel_execution=True,
#     #                                           n_cpu=12
#     #                                           )
#     #
#     ## run optmization
#     #t_1 = time.time()
#     #x_opt,y_opt,_,_,_ = layout_optimization.optimize()
#     #t_2 = time.time()
#     #aep_generations = layout_optimization.fitness_val_array_gen
#     #
#     #print(aep_generations)
#     #print(t_2-t_1)
    





