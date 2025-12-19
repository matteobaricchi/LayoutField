#%%
# import packages

import numpy as np
import matplotlib.pyplot as plt

from metrics_layout import LayoutProbabilityGrid



#%%
# define a 10 layouts of 3x3 farm in squared area with 5D min distances

x_boundaries = np.array([0.,10.,10.,0.,0.])
y_boundaries = np.array([0.,0.,10.,10.,0.])
x = np.array([0.,5.,10.,0.,5.,10.,0.,5.,10.])
y = np.array([0.,0.,0.,5.,5.,5.,10.,10.,10.])

savefig = False
name_path = 'figures_TORQUE_paper'
name_format = 'pdf'
figsize = (4,4)
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']


namefig = 'example_layout0'


def plot_layout(x,y,x_boundaries,y_boundaries,figsize,savefig,name_path,name_format,namefig,pad=2,color=colors[4],title=None):
    plt.figure(figsize=figsize)
    if title!=None: plt.title(title)
    plt.plot(x_boundaries,y_boundaries,c='k',zorder=1)
    plt.scatter(x,y,c=colors[1],marker='1',s=500,zorder=2,linewidths=3)
    plt.scatter(x,y,c=colors[1],marker='.',s=200,zorder=2)
    plt.xlabel(r'X coordinate')
    plt.ylabel(r'Y coordinate')
    plt.xticks([])
    plt.yticks([])
    plt.scatter([-1,11],[-1,11],c='white',zorder=0) # to ensure xlim and ylim preserving aspect ratio
    plt.axis('equal')
    if savefig: plt.savefig(name_path+'\\'+namefig+'.'+name_format,format=name_format,bbox_inches='tight')
    plt.show()

plot_layout(x,y,x_boundaries,y_boundaries,figsize,savefig,name_path,name_format,namefig)

#%%
# 3D plot for LF defintion

from mpl_toolkits.mplot3d import Axes3D

# define object and sigma values
layout_p_grid = LayoutProbabilityGrid(x_boundaries=x_boundaries, y_boundaries=y_boundaries, diameter=1.,res_D=0.1)

savefig = False
name_path = 'figures_TORQUE_paper'
name_format = 'pdf'
figsize = (6,4)

namefig = 'example_LF_3D_layout0_sigma15'
cmap = 'Purples'
sigma_D = 1.5

fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection='3d')
f = layout_p_grid.calculate_probability_grid([x],[y],sigma_D)
surf = ax.plot_surface(layout_p_grid.x_grid,layout_p_grid.y_grid,f.T,cmap=cmap,edgecolor='none',antialiased=True)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'X coordinate', labelpad=0)
ax.set_ylabel(r'Y coordinate', labelpad=0)
ax.set_zlabel(r'Layout field $\left[\mathrm{m}^{-2}\right]$', labelpad=5)
ax.set_box_aspect([1, 1, 0.6])
ax.view_init(elev=20, azim=135)
if savefig: plt.savefig(name_path+'\\'+namefig+'.'+name_format,format=name_format,bbox_inches=None,pad_inches=0.3)
plt.show()



#%%
# effect of sigma

savefig = False
name_path = 'figures_TORQUE_paper'
name_format = 'pdf'
figsize = (3.5,3)
max_digits = None
n_ticks = None

# plot examples for different sigma values

sigma_D = 0.5
namefig = 'example_LF_layout0_sigma05'
layout_p_grid.plot_probability_grid([x],[y],sigma_D=sigma_D,savefig=savefig,pathfig=name_path,formatfig=name_format,namefig=namefig,include_xy_labels=False,figsize=figsize,cmap='Purples',max_digits=max_digits,n_ticks=n_ticks)

sigma_D = 1.5
namefig = 'example_LF_layout0_sigma15'
layout_p_grid.plot_probability_grid([x],[y],sigma_D=sigma_D,savefig=savefig,pathfig=name_path,formatfig=name_format,namefig=namefig,include_xy_labels=False,figsize=figsize,cmap='Purples',max_digits=max_digits,n_ticks=n_ticks)

sigma_D = 2.5
namefig = 'example_LF_layout0_sigma25'
layout_p_grid.plot_probability_grid([x],[y],sigma_D=sigma_D,savefig=savefig,pathfig=name_path,formatfig=name_format,namefig=namefig,include_xy_labels=False,figsize=figsize,cmap='Purples',max_digits=max_digits,n_ticks=n_ticks)




#%%
# superimpose different layouts and layout field error (LFE)

x_min = np.min(x_boundaries)
x_max = np.max(x_boundaries)
y_min = np.min(y_boundaries)
y_max = np.max(y_boundaries)

step_min = 1.
step_max = 2.5

x_1 = np.minimum(np.maximum(x+((step_max-step_min)*np.random.rand(len(x))+step_min)*np.cos(2*np.pi*np.random.rand(len(x))),x_min),x_max)
y_1 = np.minimum(np.maximum(y+((step_max-step_min)*np.random.rand(len(x))+step_min)*np.random.rand(len(y))*np.cos(2*np.pi*np.random.rand(len(y))),y_min),y_max)

x_2 = np.minimum(np.maximum(x+((step_max-step_min)*np.random.rand(len(x))+step_min)*np.cos(2*np.pi*np.random.rand(len(x))),x_min),x_max)
y_2 = np.minimum(np.maximum(y+((step_max-step_min)*np.random.rand(len(x))+step_min)*np.random.rand(len(y))*np.cos(2*np.pi*np.random.rand(len(y))),y_min),y_max)

x_3 = np.minimum(np.maximum(x+((step_max-step_min)*np.random.rand(len(x))+step_min)*np.cos(2*np.pi*np.random.rand(len(x))),x_min),x_max)
y_3 = np.minimum(np.maximum(y+((step_max-step_min)*np.random.rand(len(x))+step_min)*np.random.rand(len(y))*np.cos(2*np.pi*np.random.rand(len(y))),y_min),y_max)

x_4 = np.minimum(np.maximum(x+((step_max-step_min)*np.random.rand(len(x))+step_min)*np.cos(2*np.pi*np.random.rand(len(x))),x_min),x_max)
y_4 = np.minimum(np.maximum(y+((step_max-step_min)*np.random.rand(len(x))+step_min)*np.random.rand(len(y))*np.cos(2*np.pi*np.random.rand(len(y))),y_min),y_max)


savefig = False
name_path = 'figures_TORQUE_paper'
name_format = 'svg'
figsize = (5.5,5)

sigma_D = 1.5

#%%
# plot superposition -------------------------------------------------------------------------

savefig = False
name_path = 'figures_TORQUE_paper'
name_format = 'pdf'
figsize = (11, 5)

namefig = 'example_superposition_sigma15'
cmap = 'Purples'

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[0.95, 1, 2.])

# Top-left
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title(r'Layout 1')
ax1.plot(x_boundaries,y_boundaries,c='k',zorder=1)
ax1.scatter(x_1,y_1,c=colors[2],marker='1',s=200,zorder=2,linewidths=2)
ax1.scatter(x_1,y_1,c=colors[2],marker='.',s=100,zorder=2)
ax1.set_xlabel(r'X coordinate')
ax1.set_ylabel(r'Y coordinate')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.scatter([-0.5,10.5],[-0.5,10.5],c='white',zorder=0) # to ensure xlim and ylim preserving aspect ratio
ax1.set_aspect('equal')

# Top-middle
ax2 = fig.add_subplot(gs[0, 1])
f = layout_p_grid.calculate_probability_grid([x_1],[y_1],sigma_D)
ax2.set_title(r'Layout 1')
im = ax2.imshow(f.T,cmap=cmap,aspect='equal',origin='lower',extent=[np.min(layout_p_grid.x_grid),np.max(layout_p_grid.x_grid),np.min(layout_p_grid.y_grid),np.max(layout_p_grid.y_grid)])
ax2.plot(layout_p_grid.x_boundaries,layout_p_grid.y_boundaries,c='k')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel(r'X coordinate')
ax2.set_ylabel(r'Y coordinate')
cbar = fig.colorbar(im,ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label(r'Layout field $\left[\mathrm{m}^{-2}\right]$')

# Bottom-left
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title(r'Layout 2')
ax3.plot(x_boundaries,y_boundaries,c='k',zorder=1)
ax3.scatter(x_2,y_2,c=colors[1],marker='1',s=200,zorder=2,linewidths=2)
ax3.scatter(x_2,y_2,c=colors[1],marker='.',s=100,zorder=2)
ax3.set_xlabel(r'X coordinate')
ax3.set_ylabel(r'Y coordinate')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.scatter([-0.5,10.5],[-0.5,10.5],c='white',zorder=0) # to ensure xlim and ylim preserving aspect ratio
ax3.set_aspect('equal')

# Bottom-middle
ax4 = fig.add_subplot(gs[1, 1])
f = layout_p_grid.calculate_probability_grid([x_2],[y_2],sigma_D)
ax4.set_title(r'Layout 2')
im = ax4.imshow(f.T,cmap=cmap,aspect='equal',origin='lower',extent=[np.min(layout_p_grid.x_grid),np.max(layout_p_grid.x_grid),np.min(layout_p_grid.y_grid),np.max(layout_p_grid.y_grid)])
ax4.plot(layout_p_grid.x_boundaries,layout_p_grid.y_boundaries,c='k')
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_xlabel(r'X coordinate')
ax4.set_ylabel(r'Y coordinate')
cbar = fig.colorbar(im,ax=ax4, fraction=0.046, pad=0.04)
cbar.set_label(r'Layout field $\left[\mathrm{m}^{-2}\right]$')

# Right column spanning both rows
ax5 = fig.add_subplot(gs[:, 2])
f = layout_p_grid.calculate_probability_grid([x_1,x_2],[y_1,y_2],sigma_D)
ax5.set_title(r'Superposition of Layouts 1 and 2')
im = ax5.imshow(f.T,cmap=cmap,aspect='equal',origin='lower',extent=[np.min(layout_p_grid.x_grid),np.max(layout_p_grid.x_grid),np.min(layout_p_grid.y_grid),np.max(layout_p_grid.y_grid)])
ax5.plot(layout_p_grid.x_boundaries,layout_p_grid.y_boundaries,c='k')
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_xlabel(r'X coordinate')
ax5.set_ylabel(r'Y coordinate')
cbar = fig.colorbar(im,ax=ax5, fraction=0.046, pad=0.04)
cbar.set_label(r'Layout field $\left[\mathrm{m}^{-2}\right]$')

plt.tight_layout()
if savefig: plt.savefig(name_path+'\\'+namefig+'.'+name_format,format=name_format,bbox_inches='tight')
plt.show()


#%%
# plot layout field difference (LFE) ---------------------------------------------------------


import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

cmap_blues = mcolors.LinearSegmentedColormap.from_list(f'reds_cmap',plt.cm.coolwarm(np.flip(np.linspace(0,0.5,100))))
cmap_reds = mcolors.LinearSegmentedColormap.from_list(f'blues_camp',plt.cm.coolwarm(np.linspace(0.5,1,100)))


savefig = False
name_path = 'figures_TORQUE_paper'
name_format = 'pdf'
figsize = (11, 5)

namefig = 'example_layout_field_error_sigma15'


fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[0.95, 1, 2.])

# Top-left
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title(r'Layouts 1 and 2')
ax1.plot(x_boundaries,y_boundaries,c='k',zorder=1)
ax1.scatter(x_1,y_1,c=colors[2],marker='1',s=200,zorder=2,linewidths=2)
ax1.scatter(x_1,y_1,c=colors[2],marker='.',s=100,zorder=2)
ax1.scatter(x_2,y_2,c=colors[1],marker='1',s=200,zorder=2,linewidths=2)
ax1.scatter(x_2,y_2,c=colors[1],marker='.',s=100,zorder=2)
ax1.set_xlabel(r'X coordinate')
ax1.set_ylabel(r'Y coordinate')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.scatter([-0.5,10.5],[-0.5,10.5],c='white',zorder=0) # to ensure xlim and ylim preserving aspect ratio
ax1.set_aspect('equal')

# Top-middle
ax2 = fig.add_subplot(gs[0, 1])
f = layout_p_grid.calculate_probability_grid([x_1,x_2],[y_1,y_2],sigma_D)
ax2.set_title(r'Layout field 1+2')
cmap = cmap_blues
im = ax2.imshow(f.T,cmap=cmap,aspect='equal',origin='lower',extent=[np.min(layout_p_grid.x_grid),np.max(layout_p_grid.x_grid),np.min(layout_p_grid.y_grid),np.max(layout_p_grid.y_grid)])
ax2.plot(layout_p_grid.x_boundaries,layout_p_grid.y_boundaries,c='k')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel(r'X coordinate')
ax2.set_ylabel(r'Y coordinate')
cbar = fig.colorbar(im,ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label(r'Layout field $\left[\mathrm{m}^{-2}\right]$')

# Bottom-left
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title(r'Layouts 3 and 4')
ax3.plot(x_boundaries,y_boundaries,c='k',zorder=1)
ax3.scatter(x_3,y_3,c=colors[4],marker='1',s=200,zorder=2,linewidths=2)
ax3.scatter(x_3,y_3,c=colors[4],marker='.',s=100,zorder=2)
ax3.scatter(x_4,y_4,c=colors[3],marker='1',s=200,zorder=2,linewidths=2)
ax3.scatter(x_4,y_4,c=colors[3],marker='.',s=100,zorder=2)
ax3.set_xlabel(r'X coordinate')
ax3.set_ylabel(r'Y coordinate')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.scatter([-0.5,10.5],[-0.5,10.5],c='white',zorder=0) # to ensure xlim and ylim preserving aspect ratio
ax3.set_aspect('equal')

# Bottom-middle
ax4 = fig.add_subplot(gs[1, 1])
f = layout_p_grid.calculate_probability_grid([x_3,x_4],[y_3,y_4],sigma_D)
ax4.set_title(r'Layout field 3+4')
cmap = cmap_reds
im = ax4.imshow(f.T,cmap=cmap,aspect='equal',origin='lower',extent=[np.min(layout_p_grid.x_grid),np.max(layout_p_grid.x_grid),np.min(layout_p_grid.y_grid),np.max(layout_p_grid.y_grid)])
ax4.plot(layout_p_grid.x_boundaries,layout_p_grid.y_boundaries,c='k')
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_xlabel(r'X coordinate')
ax4.set_ylabel(r'Y coordinate')
cbar = fig.colorbar(im,ax=ax4, fraction=0.046, pad=0.04)
cbar.set_label(r'Layout field $\left[\mathrm{m}^{-2}\right]$')

# Right column spanning both rows
ax5 = fig.add_subplot(gs[:, 2])
f_diff = layout_p_grid.calculate_diff_grid([x_3,x_4],[y_3,y_4],[x_1,x_2],[y_1,y_2],sigma_D)
halfrange = np.maximum(-np.min(f_diff),np.max(f_diff))
ax5.set_title(r'Layout field error $(3+4)-(1+2)$')
cmap = 'coolwarm'
im = ax5.imshow(f_diff.T,cmap=cmap,norm=mcolors.CenteredNorm(halfrange=halfrange),aspect='equal',origin='lower',extent=[np.min(layout_p_grid.x_grid),np.max(layout_p_grid.x_grid),np.min(layout_p_grid.y_grid),np.max(layout_p_grid.y_grid)])
ax5.plot(layout_p_grid.x_boundaries,layout_p_grid.y_boundaries,c='k')
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_xlabel(r'X coordinate')
ax5.set_ylabel(r'Y coordinate')
cbar = fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
cbar.set_label(r'Layout field error $\left[\mathrm{m}^{-2}\right]$')

plt.tight_layout()
if savefig: plt.savefig(name_path+'\\'+namefig+'.'+name_format,format=name_format,bbox_inches='tight')
plt.show()


