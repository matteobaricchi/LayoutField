# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:32:25 2025

@author: matteobaricchi
"""

import numpy as np
from numpy import newaxis as na

import matplotlib.pyplot as plt
import matplotlib.colors as colors


class LayoutProbabilityGrid():
    
    # the values of f_grid_tot are dependent on the resolution. It is valid: np.sum(f)*(self.res_D*self.diameter)**2 if res_D is small enough
    
    def __init__(self,
                 x_boundaries,
                 y_boundaries,
                 diameter,
                 res_D = 0.1    # [-] expressed as a fraction of the diameter
                 ):
        
        self.x_boundaries = x_boundaries
        self.y_boundaries = y_boundaries
        self.diameter = diameter
        self.res_D = res_D
        
        # calculate x and y coord grids
        x_min = np.min(self.x_boundaries) - 0.1*(np.max(self.x_boundaries)-np.min(self.x_boundaries))
        x_max = np.max(self.x_boundaries) + 0.1*(np.max(self.x_boundaries)-np.min(self.x_boundaries))
        y_min = np.min(self.y_boundaries) - 0.1*(np.max(self.y_boundaries)-np.min(self.y_boundaries))
        y_max = np.max(self.y_boundaries) + 0.1*(np.max(self.y_boundaries)-np.min(self.y_boundaries))
        n_x = int(np.round((x_max-x_min)/(self.res_D*self.diameter)))
        n_y = int(np.round((y_max-y_min)/(self.res_D*self.diameter)))
        x_values = np.linspace(x_min,x_max,n_x)
        y_values = np.linspace(y_min,y_max,n_y)
        self.x_grid = np.tile(x_values[:,na],(1,len(y_values)))
        self.y_grid = np.tile(y_values[na,:],(len(x_values),1))
        

    def _calculate_probability_grid(self,x_layout,y_layout,sigma_D):
        sigma = sigma_D*self.diameter
        f_grid = (1/(2*np.pi*sigma**2))*np.exp(-((self.x_grid[:,:,na]-x_layout[na,na,:])**2+(self.y_grid[:,:,na]-y_layout[na,na,:])**2)/(2*sigma**2))
        f_grid_tot = np.sum(f_grid,axis=(2))
        # f_grid_tot_norm_temp = f_grid_tot*((self.res_D*self.diameter)**2)                       # normalization (to ensure that sum(f)=n_wt) -> sometimes not exactly
        # f_grid_tot_norm = f_grid_tot_norm_temp*(len(x_layout)/np.sum(f_grid_tot_norm_temp))     # explicit normalization to be sure that sum(f)=n_wt
        return f_grid_tot
    
    def calculate_probability_grid(self,x_list,y_list,sigma_D):
        # this function averages the superimposition of multiple layouts
        f_all = np.tile(np.zeros_like(self.x_grid)[:,:,na],(1,1,len(x_list)))
        for i in np.arange(len(x_list)): f_all[:,:,i] = self._calculate_probability_grid(x_list[i],y_list[i],sigma_D)
        f_mean = np.mean(f_all,axis=(2))
        return f_mean
    
    def plot_probability_grid(self,
                              x_list,
                              y_list,
                              sigma_D,          # [-] expressed as a fraction of the diameter
                              figsize = (8,8),
                              title = '', 
                              cmap = 'Blues',
                              savefig = False,
                              pathfig = '',
                              namefig ='',
                              formatfig = '',
                              include_xy_labels = True,
                              max_digits = None,
                              n_ticks = None
                              ):
        
        f = self.calculate_probability_grid(x_list,y_list,sigma_D)
        
        fig,ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        im = ax.imshow(f.T,cmap=cmap,aspect='equal',origin='lower',extent=[np.min(self.x_grid),np.max(self.x_grid),np.min(self.y_grid),np.max(self.y_grid)])
        ax.plot(self.x_boundaries,self.y_boundaries,c='k')
        if include_xy_labels:
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'X coordinate')
            ax.set_ylabel(r'Y coordinate')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'Layout field $\left[\mathrm{m}^{-2}\right]$')
        if (max_digits!=None)&(n_ticks!=None):
            #from matplotlib.ticker import FormatStrFormatter
            vmin, vmax = im.get_clim()
            ticks = np.linspace(vmin, vmax, n_ticks)
            cbar.set_ticks(ticks)
            #cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{max_digits}f'))
            cbar.update_ticks()
            labels = [f"{t:.{max_digits}f}" for t in ticks]
            cbar.set_ticklabels(labels)

        if savefig: plt.savefig(pathfig+'\\'+namefig+'.'+formatfig,format=formatfig,bbox_inches='tight')
        plt.show()
        
        
    
    def calculate_diff_grid(self,x_list_1,y_list_1,x_list_2,y_list_2,sigma_D):
        f_1 = self.calculate_probability_grid(x_list_1,y_list_1,sigma_D)
        f_2 = self.calculate_probability_grid(x_list_2,y_list_2,sigma_D)
        return f_1-f_2
    
    def plot_diff_grid(self,
                       x_list_1,
                       y_list_1,
                       x_list_2,
                       y_list_2,
                       sigma_D,          # [-] expressed as a fraction of the diameter
                       halfrange = None,
                       figsize = (8,8),
                       title = '', 
                       cmap = 'coolwarm',
                       savefig = False,
                       pathfig = '',
                       namefig ='',
                       formatfig = '',
                       include_xy_labels = True
                       ):
        
        f_diff = self.calculate_diff_grid(x_list_1,y_list_1,x_list_2,y_list_2,sigma_D)
        
        if halfrange == None: halfrange = np.maximum(-np.min(f_diff),np.max(f_diff))
                
        fig,ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        im = ax.imshow(f_diff.T,cmap=cmap,norm=colors.CenteredNorm(halfrange=halfrange),aspect='equal',origin='lower',extent=[np.min(self.x_grid),np.max(self.x_grid),np.min(self.y_grid),np.max(self.y_grid)])
        ax.plot(self.x_boundaries,self.y_boundaries,c='k')
        if include_xy_labels:
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'X coordinate')
            ax.set_ylabel(r'Y coordinate')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'Layout field $\left[\mathrm{m}^{-2}\right]$')
        if savefig: plt.savefig(pathfig+'\\'+namefig+'.'+formatfig,format=formatfig,bbox_inches='tight')
        plt.show()
            
    
    def calculate_integrated_diff(self,x_list_1,y_list_1,x_list_2,y_list_2,sigma_D):
        f_diff = self.calculate_diff_grid(x_list_1,y_list_1,x_list_2,y_list_2,sigma_D)
        return np.sum(np.abs(f_diff))*(self.res_D*self.diameter)**2

