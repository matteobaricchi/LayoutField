#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import pandas as pd
from numpy import newaxis as na
import math
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D


from metrics_layout.metrics_layout import LayoutProbabilityGrid

from scipy.special import erfinv

import matplotlib.transforms as mtransforms





#%%
# functions

def from_lfie_to_d_eff(lfie,sigma_D,n_wt=69):
    return 2*np.sqrt(2)*sigma_D*erfinv(lfie/(2*69))

def plot_matrix_lfie_v3(mat,
                        fig,
                        ax,
                        mat_filter=None,
                        cmap='Blues', # list of len(cmap_bouds)+1 elements 
                        title='',
                        cases_list=None,
                        pad=10,
                        sizes=[0.3,0.6,0.9],
                        size_labels = ['Low', 'Moderate', 'High'],
                        cmap_bounds=None,
                        cbar_show_referent=True,
                        cbar_label='LFIE [-]',
                        cbar_ticklabels=None,
                        direct_comparison_mat=None):
    
    norm = mcolors.Normalize(vmin=mat.min(),vmax=mat.max())
    cmap = plt.get_cmap(cmap)
     
    if mat_filter is not None:
        mat[~mat_filter] = np.nan

    ax.set_title(title)
    ax.set_xlim(-0.5,mat.shape[1]-0.5)
    ax.set_ylim(-0.5,mat.shape[1]-0.5)
    ax.set_xticks(range(len(cases_list)))
    ax.set_yticks(np.flip(range(len(cases_list))))
    ax.set_xticklabels(cases_list)
    ax.set_yticklabels(cases_list)
    ax.set_xlabel('Design choice')
    ax.set_ylabel('Design choice')
    ax.tick_params(axis='x', pad=pad)
    ax.tick_params(axis='y', pad=pad)
    plt.setp(ax.get_xticklabels(), ha="right", va="top",rotation=60)
    plt.setp(ax.get_yticklabels(), ha="right", va="center", rotation=0)


    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            bin_idx = np.digitize(mat[i,j],cmap_bounds)
            s = sizes[bin_idx]
            x = j
            y = mat.shape[0] - 1 - i
            rect = plt.Rectangle((x-s/2,y-s/2),s,s,facecolor=cmap(norm(mat[i,j])),edgecolor='white',linewidth=3)
            ax.add_patch(rect)
            if ~direct_comparison_mat[i,j]:
                rect = plt.Rectangle((x-0.5,y-0.5),1,1,fill=False,hatch='/////////',edgecolor='white',linewidth=5)
                ax.add_patch(rect)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label)
    if cbar_show_referent:
        cbar.set_ticks(cmap_bounds)
        cbar.set_ticklabels(cbar_ticklabels)

    legend_handles = [Line2D([0],[0],marker='s',markersize=1+15*s,linestyle='',markerfacecolor='gray',markeredgecolor='white') for s in sizes]
    leg1 = ax.legend(legend_handles,size_labels,title='Impact',loc='lower center',bbox_to_anchor=(0.3, 1.02),ncol=len(sizes))
    ax.add_artist(leg1)
    scale = 1+15*np.max(sizes)
    hatch_legend = [patches.Rectangle((0, 0),scale,scale,facecolor='gray',edgecolor='white',label='Direct'),patches.Rectangle((0, 0),scale,scale,facecolor='gray',edgecolor='white',hatch='/////////',label='Indirect')]
    leg2 = ax.legend(handles=hatch_legend,title='Comparison',loc='lower center',bbox_to_anchor=(0.85, 1.02),ncol=2,handlelength=1.0,handleheight=1.0)




def plot_matrix_lfie_v4(mat,
                        fig,
                        ax,
                        mat_filter=None,
                        color='b',
                        alpha_indirect=0.5,
                        title='',
                        cases_list=None,
                        pad=10,
                        sizes=[0.2,0.5,0.8],
                        size_labels = ['Low', 'Moderate', 'High'],
                        cmap_bounds=None,
                        direct_comparison_mat=None):
         
    if mat_filter is not None:
        mat[~mat_filter] = np.nan

    ax.set_title(title)
    ax.set_xlim(-0.5,mat.shape[1]-0.5)
    ax.set_ylim(-0.5,mat.shape[1]-0.5)
    ax.set_xticks(range(len(cases_list)))
    ax.set_yticks(np.flip(range(len(cases_list))))
    ax.set_xticklabels(cases_list)
    ax.set_yticklabels(cases_list)
    ax.set_xlabel('Design choice')
    ax.set_ylabel('Design choice')
    ax.tick_params(axis='x', pad=pad)
    ax.tick_params(axis='y', pad=pad)
    plt.setp(ax.get_xticklabels(), ha="right", va="top",rotation=60)
    plt.setp(ax.get_yticklabels(), ha="right", va="center", rotation=0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            bin_idx = np.digitize(mat[i,j],cmap_bounds)
            s = sizes[bin_idx]
            if mat[i,j]<1e-6:
                s = 0
            x = j
            y = mat.shape[0] - 1 - i
            if direct_comparison_mat[i,j]:
                rect = plt.Rectangle((x-s/2,y-s/2),s,s,facecolor=color,linewidth=3)
            else:
                rect = plt.Rectangle((x-s/2,y-s/2),s,s,facecolor=color,linewidth=3,alpha=alpha_indirect)
            ax.add_patch(rect)

    legend_handles = [Line2D([0],[0],marker='s',markersize=1+15*s,linestyle='',markerfacecolor='gray',markeredgecolor='gray') for s in sizes]
    leg1 = ax.legend(legend_handles,size_labels,title='Impact',loc='lower center',bbox_to_anchor=(0.3, 1.02),ncol=len(sizes))
    ax.add_artist(leg1)
    scale = 1+15*np.max(sizes)
    hatch_legend = [patches.Rectangle((0, 0),scale,scale,facecolor=color,label='Direct'),patches.Rectangle((0, 0),scale,scale,facecolor=color,edgecolor='white',label='Indirect',alpha=alpha_indirect)]
    leg2 = ax.legend(handles=hatch_legend,title='Comparison',loc='lower center',bbox_to_anchor=(0.85, 1.02),ncol=2,handlelength=1.0,handleheight=1.0)




def plot_matrix_lfie_sensAn_v4(mat,
                               fig,
                               ax,
                               color='b',
                               title='',
                               cases_list=None,
                               rowname_list=None,
                               pad=10,
                               sizes=[0.2,0.5,0.8],
                               size_labels = ['Low', 'Moderate', 'High'],
                               cmap_bounds_list=None):
         

    ax.set_title(title)
    ax.set_xlim(-0.5,mat.shape[1]-0.5)
    ax.set_ylim(-0.5,mat.shape[0]-0.5)
    ax.set_xticks(range(len(cases_list)))
    ax.set_yticks(np.flip(range(len(rowname_list))))
    ax.set_xticklabels(cases_list)
    ax.set_yticklabels(rowname_list)
    ax.set_xlabel('Design choice')
    #ax.set_ylabel('Case study')
    ax.tick_params(axis='x', pad=pad)
    ax.tick_params(axis='y', pad=pad)
    plt.setp(ax.get_xticklabels(), ha="right", va="top",rotation=60)
    plt.setp(ax.get_yticklabels(), ha="right", va="center", rotation=0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            bin_idx = np.digitize(mat[i,j],cmap_bounds_list[i])
            s = sizes[bin_idx]
            if mat[i,j]<1e-6:
                s = 0
            x = j
            y = mat.shape[0] - 1 - i

            rect = plt.Rectangle((x-s/2,y-s/2),s,s,facecolor=color,linewidth=3)
            ax.add_patch(rect)

    legend_handles = [Line2D([0],[0],marker='s',markersize=1+15*s,linestyle='',markerfacecolor='gray',markeredgecolor='gray') for s in sizes]
    leg1 = ax.legend(legend_handles,size_labels,title='Impact',loc='lower center',bbox_to_anchor=(0.5, 1.02),ncol=len(sizes))






# create filter matrix
def create_category_filter(category_list_obj,category_list_eval):
    mat_filter = np.zeros((len(category_list_eval),len(category_list_obj)),dtype=bool)
    for i in np.arange(len(category_list_eval)):
        for j in np.arange(len(category_list_obj)):
            mat_filter[i,j] = category_list_eval[i]==category_list_obj[j]
    mat_filter[category_list_eval.index('Referent'),:] = True
    mat_filter[:,category_list_obj.index('Referent')] = True
    return mat_filter

def assign_color_to_category(category_list,colors):
    color_list = [None]*len(category_list)
    c_ind = 0
    color_list[0] = colors[c_ind]
    for i in np.arange(1,len(color_list)):
        if category_list[i-1]!=category_list[i]:
            c_ind += 1
        color_list[i] = colors[c_ind]
    return color_list



# function to create dictionary
def create_dict_case(n_samples,name_path,namefile_root,f_name_list):
    dict_case = {
        'x_list' : [None]*len(n_samples),
        'y_list' : [None]*len(n_samples)
    }
    for f_name in f_name_list:
        dict_case[f_name] = np.ones(len(n_samples))*np.nan
    for i in np.arange(len(n_samples)):
        name_file = namefile_root+f'{n_samples[i]}.pkl'
        with open(name_path+name_file,'rb') as f:
            data = pickle.load(f)
            dict_case['x_list'][i] = data['x_opt']
            dict_case['y_list'][i] = data['y_opt']
            for f_name in f_name_list:
                dict_case[f_name][i] = data.get(f_name, data.get(f_name + '()'))#data[f_name]    
    return dict_case


#%%
# extract data

dataname_list = [
    'AEPTurbOPark',
    'AEPJensen',
    'AEPBastankhah',
    'AEPZong',
    'AEPTurbOParkLsum',
    'AEPTurbOParkRcenter',
    'AEPTurbOParkCGI',
    'AEPTurbOPark5deg',
    'AEPTurbOPark10deg',
    'AEPTurbOPark8ms',
    'AEPTurbOParkaWR',
    'LCOETurbOPark',
    #'IOETurbOPark',
    'REVTurbOPark',
    'COVETurbOPark',
    'AEPTurbOParkGY',
    'AEPTurbOParkCab',
    'AEPTurbOParkSGD',
    'AEPTurbOParkLOGAf',
    ]

# extract LFIE
n_samples = np.array([1,2,3,4,5,6,7,8,9,10])
name_path = 'data_LF/'
root_name = 'HKNscaled_'
data_LFIE = {}
for i in np.arange(len(dataname_list)):
    data_LFIE[dataname_list[i]] = {}
    with open(name_path+root_name+dataname_list[i]+'/'+root_name+dataname_list[i]+'_LFIE'+'.pkl', 'rb') as f:
        dict_LFIE = pickle.load(f)
    dict_LFIE_s = dict_LFIE['dict_LFIE_s']
    for j in np.arange(len(dataname_list)):
        data_LFIE[dataname_list[i]][dataname_list[j]] = dict_LFIE_s[root_name+dataname_list[j]]

# extract sigma array
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_LFIE'+'.pkl', 'rb') as f:
    dict_LFIE = pickle.load(f)
sigma_D_array = dict_LFIE['sigma_D_array']

# extract referent self comparison
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_self_LFIE'+'.pkl', 'rb') as f:
    dict_self_LFIE = pickle.load(f)
referent_self_lfie_ns = dict_self_LFIE['LFIE_ns']

# extract referent comparison
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_LFIE'+'.pkl', 'rb') as f:
    dict_LFIE = pickle.load(f)
dict_LFIE_s = dict_LFIE['dict_LFIE_s']
referent_lfie_s = dict_LFIE_s[root_name+'AEPTurbOPark2']


label_list = [
    'REF',
    'wdm-JENS',
    'wdm-BAST',
    'wdm-ZONG',
    'wsm-LSUM',
    'ram-RCEN',
    'ram-CGI',
    'wdd-5DEG',
    'wdd-10DEG',
    'wsd-1VAL',
    'wrd-HOM',
    'obj-LCOE',
    #'obj-IOE',
    'obj-REV',
    'obj-COVE',
    'cod-YAW',
    'cod-CAB',
    'opt-SGD',
    'opt-LOGAf',
]


category_list = [
    'Referent',
    'Wake deficit model',
    'Wake deficit model',
    'Wake deficit model',
    'Wake superposition model',
    'Rotor average model',
    'Rotor average model',
    'Wind direction discretization',
    'Wind direction discretization',
    'Wind speed discretization',
    'Wind resource data',
    'Objective function choice',
    #'Objective function choice',
    'Objective function choice',
    'Objective function choice',
    'Wind farm co-design',
    'Wind farm co-design',
    'Optimization algorithm',
    'Optimization algorithm',
    ]

category_list_unique = list(dict.fromkeys(category_list))


# %%
# plot matrices for a given sigma

ind_sigma = 2

lfie_mat = np.ones((len(dataname_list),len(dataname_list)))*np.nan
for i in np.arange(len(dataname_list)):
    for j in np.arange(len(dataname_list)):
        lfie_mat[i,j] = data_LFIE[dataname_list[i]][dataname_list[j]][ind_sigma]


savefig = False
name_path = '../figures'
name_format = 'pdf'

# d eff mat
mat_filter = create_category_filter(category_list,category_list)
referent_self_d_eff = from_lfie_to_d_eff(np.median(referent_self_lfie_ns[:,ind_sigma],axis=(0)),sigma_D_array[ind_sigma])
referent_d_eff = from_lfie_to_d_eff(referent_lfie_s[ind_sigma],sigma_D_array[ind_sigma])
fig, ax = plt.subplots(figsize=(8.5,8))
plot_matrix_lfie_v4(from_lfie_to_d_eff(lfie_mat,sigma_D_array[ind_sigma]),fig,ax,cases_list=label_list,cmap_bounds=[referent_d_eff,referent_self_d_eff],direct_comparison_mat=mat_filter,color=plt.get_cmap('Blues')(0.95),alpha_indirect=0.2,sizes=[0.2,0.4,0.85])
if savefig: plt.savefig(name_path+'\\'+'deff_mat_s5_v4'+'.'+name_format,format=name_format,bbox_inches='tight')
plt.show()



# %%
# plot barplot for a given sigma (zoom in on the first row/col of the matrix)

ind_sigma = 2

lfie_array = np.ones((len(dataname_list)))*np.nan
for i in np.arange(len(dataname_list)):
    lfie_array[i] = data_LFIE[dataname_list[0]][dataname_list[i]][ind_sigma]


savefig = False
name_path = '../figures'
name_format = 'pdf'


# calculate d_eff values
d_eff_array = from_lfie_to_d_eff(lfie_array,sigma_D_array[ind_sigma])
referent_self_d_eff = from_lfie_to_d_eff(np.median(referent_self_lfie_ns[:,ind_sigma],axis=(0)),sigma_D_array[ind_sigma])
referent_d_eff = from_lfie_to_d_eff(referent_lfie_s[ind_sigma],sigma_D_array[ind_sigma])

# define colors
colors = ['#000000', "#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#bbbbbb", "#ffa500", "#44aa99" ]
colors_plot = [colors[0]]*1+[colors[1]]*3+[colors[2]]*1+[colors[3]]*2+[colors[4]]*2+[colors[5]]*1+[colors[6]]*1+[colors[7]]*3+[colors[8]]*2+[colors[9]]*2

fig,ax = plt.subplots(figsize=(8,5))

# plot data
x_plot = np.arange(len(d_eff_array))
ax.bar(x_plot,d_eff_array,color=colors_plot)
ax.axhline(referent_d_eff,c='k',linestyle='-')
ax.axhline(referent_self_d_eff,c='k',linestyle='--')
ax.set_xlim([np.min(x_plot)-0.5,np.max(x_plot)+0.5])

# set labels
ax.set_xticks(x_plot)
ax.set_xticklabels(label_list,rotation=45,ha='right')
ax.set_xlabel('Design choice')
ax.set_ylabel('Effective layout distance [D]')
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks([referent_d_eff,referent_self_d_eff])
ax2.set_yticklabels(['REFvsREF','self REF'])

handles2 = [ ax.scatter([],[],marker='s',color=colors[i],label=category_list_unique[i],edgecolor='k') for i in np.arange(len(category_list_unique))]
legend2 = ax.legend(handles=handles2,ncols=2,loc='lower center', bbox_to_anchor=(0.5,1.02))

plt.tight_layout()
if savefig: plt.savefig(name_path+'\\'+'deff_barplot_s5_v2'+'.'+name_format,format=name_format,bbox_inches='tight')
plt.show()




#%%
# sensitivty analysis: choice of sigma

ind_sigma_range = np.arange(len(sigma_D_array))

lfie_mat_sigma = np.ones((len(dataname_list),len(ind_sigma_range)))*np.nan
for i in np.arange(len(dataname_list)):
    lfie_mat_sigma[i,:] = data_LFIE[dataname_list[0]][dataname_list[i]][ind_sigma_range]

savefig = False
name_path = '../figures'
name_format = 'svg'

# calculate d_eff values
d_eff_mat_sigma = from_lfie_to_d_eff(lfie_mat_sigma,sigma_D_array[ind_sigma_range][na,:])
referent_self_d_eff_array = from_lfie_to_d_eff(np.median(referent_self_lfie_ns,axis=(0)),sigma_D_array[ind_sigma_range])
referent_d_eff_array = from_lfie_to_d_eff(referent_lfie_s,sigma_D_array[ind_sigma_range])

# define colors
colors = ['#000000', "#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#bbbbbb", "#ffa500", "#44aa99" ]
colors_plot = [colors[0]]*1+[colors[1]]*3+[colors[2]]*1+[colors[3]]*2+[colors[4]]*2+[colors[5]]*1+[colors[6]]*1+[colors[7]]*3+[colors[8]]*2+[colors[9]]*2

fig,ax = plt.subplots(figsize=(6.5,5))

# plot data
for i in np.arange(d_eff_mat_sigma.shape[0]):
    ax.plot(sigma_D_array[ind_sigma_range],d_eff_mat_sigma[i,:],c=colors_plot[i],linewidth=3,zorder=0)
ax.plot(sigma_D_array[ind_sigma_range],referent_d_eff_array,c='k',linewidth=1,linestyle='-')
ax.plot(sigma_D_array[ind_sigma_range],referent_self_d_eff_array,c='k',linewidth=1,linestyle='--')
ax.set_xlim([np.min(sigma_D_array[ind_sigma_range]),np.max(sigma_D_array[ind_sigma_range])])

# set labels
ax.set_xlabel('Smoothness parameter [D]')
ax.set_ylabel('Effective layout distance [D]')
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks([referent_d_eff_array[-1],referent_self_d_eff_array[-1]])
ax2.set_yticklabels(['REFvsREF','self REF'])

handles2 = [ ax.scatter([],[],marker='s',color=colors[i],label=category_list_unique[i],edgecolor='k') for i in np.arange(len(category_list_unique))]
legend2 = ax.legend(handles=handles2,ncols=2,loc='lower center', bbox_to_anchor=(0.5,1.02))

plt.tight_layout()
if savefig: plt.savefig(name_path+'\\'+'sAn_deff_impact_sigma_v2'+'.'+name_format,format=name_format,bbox_inches='tight')
plt.show()



#%%
# sensitivity analysis: number of layouts

# extract data

n_samples = np.array([1,2,3,4,5,6,7,8,9,10])
name_path = 'data_LF/HKNscaled_sensAn_nLayouts'
root_name = 'HKNscaled_'
cases_sensAn = ['5v5','10v10','15v15','20v20']
dataname_list_sensAn = ['AEPTurbOPark','AEPJensen','AEPTurbOParkRcenter']

# extract LFIE
data_LFIE_sensAn = {}
for i in np.arange(len(dataname_list_sensAn)):
    data_mat = np.nan*np.ones((len(cases_sensAn),len(sigma_D_array)))
    for j in np.arange(len(cases_sensAn)):
        with open(name_path+'/'+root_name+dataname_list_sensAn[i]+'_'+cases_sensAn[j]+'_LFIE'+'.pkl', 'rb') as f:
            dict_LFIE_s = pickle.load(f)
        data_mat[j,:] = dict_LFIE_s['dict_LFIE_s']['AEPTurbOPark']
    data_LFIE_sensAn[dataname_list_sensAn[i]] = data_mat

# extract self LFIE (only referent) - calculate already median
data_self_LFIE_sensAn = {}
data_mat = np.nan*np.ones((len(cases_sensAn),len(sigma_D_array)))
for j in np.arange(len(cases_sensAn)):
    with open(name_path+'/'+root_name+'AEPTurbOPark'+'_'+cases_sensAn[j]+'_self_LFIE'+'.pkl', 'rb') as f:
        dict_self_LFIE_ns = pickle.load(f)
    data_mat[j,:] = np.median(dict_self_LFIE_ns['LFIE_ns'],axis=0)
data_self_LFIE_sensAn['AEPTurbOPark'] = data_mat

# convert to effective layout distance
data_deff_sensAn = {}
data_self_deff_sensAn = {}
for i in np.arange(len(dataname_list_sensAn)):
    data_deff_sensAn[dataname_list_sensAn[i]] = from_lfie_to_d_eff(data_LFIE_sensAn[dataname_list_sensAn[i]],sigma_D_array[na,:])
data_self_deff_sensAn['AEPTurbOPark'] = from_lfie_to_d_eff(data_self_LFIE_sensAn['AEPTurbOPark'],sigma_D_array[na,:])



ind_sigma = 2

referent_d_eff_array = data_deff_sensAn['AEPTurbOPark'][:,ind_sigma]
referent_self_d_eff_array = data_self_deff_sensAn['AEPTurbOPark'][:,ind_sigma]

savefig = False
name_path = '../figures'
name_format = 'svg'

# define colors
colors = ['#000000', "#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#bbbbbb", "#ffa500", "#44aa99" ]
colors_plot = [colors[0]]+[colors[1]]+[colors[3]]

fig,ax = plt.subplots(figsize=(5,4))

# plot data
n_layouts = np.array([5,10,15,20])
for i in np.arange(1,len(dataname_list_sensAn)):
    ax.plot(n_layouts,data_deff_sensAn[dataname_list_sensAn[i]][:,ind_sigma],c=colors_plot[i],linewidth=3,zorder=0)
ax.plot(n_layouts,referent_d_eff_array,c='k',linewidth=1,linestyle='-')
ax.plot(n_layouts,referent_self_d_eff_array,c='k',linewidth=1,linestyle='--')
ax.set_xlim([np.min(n_layouts),np.max(n_layouts)])

# set labels
ax.set_xticks(n_layouts)
ax.set_xlabel('Number of layouts [-]')
ax.set_ylabel('Effective layout distance [D]')
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks([referent_d_eff_array[-1],referent_self_d_eff_array[-1],data_deff_sensAn['AEPJensen'][-1,ind_sigma],data_deff_sensAn['AEPTurbOParkRcenter'][-1,ind_sigma]]) # fixed manually
ax2.set_yticklabels(['REFvsREF','selfREF','wdm-JENS','ram-RCEN'])
labels = ax2.get_yticklabels()
labels[2].set_color(colors_plot[1])
labels[3].set_color(colors_plot[2])
labels[2].set_fontweight('bold')
labels[3].set_fontweight('bold')
labels[0].set_transform(labels[0].get_transform()+mtransforms.ScaledTranslation(0,4/72,fig.dpi_scale_trans))
#labels[2].set_transform(labels[2].get_transform()+mtransforms.ScaledTranslation(0,-21/72,fig.dpi_scale_trans))
labels[3].set_transform(labels[3].get_transform()+mtransforms.ScaledTranslation(0,-4/72,fig.dpi_scale_trans))

plt.tight_layout()
if savefig: plt.savefig(name_path+'\\'+'sAn_deff_impact_nLayouts_v2'+'.'+name_format,format=name_format,bbox_inches='tight')
plt.show()



#%%
# plot example LFE maps

# extract HKN data
import utm
with open(f'HKN_data_and_tools/HKN_data.pkl', 'rb') as f:
    HKN_data = pickle.load(f)
hkn_boundaries_x = HKN_data['hkn_boundaries_x']
hkn_boundaries_y = HKN_data['hkn_boundaries_y']
coord_sub = utm.from_latlon(52.70,4.29)
x_sub = coord_sub[0]
y_sub = coord_sub[1]
diameter = 283.2
diameter_hkn = 200.
hkn_boundaries_x_scaled = x_sub + (hkn_boundaries_x-x_sub)*(diameter/diameter_hkn)
hkn_boundaries_y_scaled = y_sub + (hkn_boundaries_y-y_sub)*(diameter/diameter_hkn)

# define object 
layout_p_grid = LayoutProbabilityGrid(x_boundaries=hkn_boundaries_x_scaled,y_boundaries=hkn_boundaries_y_scaled,diameter=diameter,res_D=0.1)
x_dim = layout_p_grid.x_grid.shape[0]
y_dim = layout_p_grid.x_grid.shape[1]
f_e_int = (layout_p_grid.res_D*layout_p_grid.diameter)**2 # correction factor to compute integrated error


ind_sigma = 2

savefig = False
name_path = '../figures'
name_format = 'svg'

sigma_D = sigma_D_array[ind_sigma]
figsize = (3.5,3)

halfrange = 15e-8 # to have the same colorbar for all plots

n_samples = np.array([1,2,3,4,5,6,7,8,9,10])
dict_AEPTurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOPark/data/',namefile_root='data_HKNscaled_AEPTurbOPark_v',f_name_list=[])

#%%
# LFE maps of wake deficit models cases

from matplotlib.colors import LinearSegmentedColormap

def make_diverging_cmap(cmap_bottom='Blues',cmap_top='Reds',n=256,center_color=(1, 1, 1, 1)):
    n_half = n // 2
    cmap_b = plt.get_cmap(cmap_bottom)
    cmap_t = plt.get_cmap(cmap_top)
    colors_bottom = cmap_b(np.linspace(1, 0, n_half))   # flipped
    colors_top    = cmap_t(np.linspace(0, 1, n_half))
    colors = np.vstack((colors_bottom,np.array([center_color]),colors_top))
    return LinearSegmentedColormap.from_list(f"{cmap_bottom}_{cmap_top}_diverging", colors)

cmap_TurbOPark = 'Reds'
cmap_Jensen = 'Blues'
cmap_Bastankhah = 'Greens'
cmap_Zong = 'Greys'


cmap = make_diverging_cmap(cmap_top=cmap_TurbOPark,cmap_bottom=cmap_Jensen)
dict_AEPJensen = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPJensen/data/',namefile_root='data_HKNscaled_AEPJensen_v',f_name_list=[])
namefig = 'LFE_AEPTurbOPark_vs_AEPJensen_s5_v2'
layout_p_grid.plot_diff_grid(dict_AEPTurbOPark['x_list'],dict_AEPTurbOPark['y_list'],dict_AEPJensen['x_list'],dict_AEPJensen['y_list'],sigma_D=sigma_D,savefig=savefig, pathfig=name_path, formatfig=name_format, namefig=namefig, include_xy_labels=False, figsize=figsize, halfrange=halfrange,cmap=cmap)

cmap = make_diverging_cmap(cmap_top=cmap_TurbOPark,cmap_bottom=cmap_Bastankhah)
dict_AEPBastankhah = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPBastankhah/data/',namefile_root='data_HKNscaled_AEPBastankhah_v',f_name_list=[])
namefig = 'LFE_AEPTurbOPark_vs_AEPBastankhah_s5_v2'
layout_p_grid.plot_diff_grid(dict_AEPTurbOPark['x_list'],dict_AEPTurbOPark['y_list'],dict_AEPBastankhah['x_list'],dict_AEPBastankhah['y_list'],sigma_D=sigma_D,savefig=savefig, pathfig=name_path, formatfig=name_format, namefig=namefig, include_xy_labels=False, figsize=figsize, halfrange=halfrange,cmap=cmap)

cmap = make_diverging_cmap(cmap_top=cmap_TurbOPark,cmap_bottom=cmap_Zong)
dict_AEPZong = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPZong/data/',namefile_root='data_HKNscaled_AEPZong_v',f_name_list=[])
namefig = 'LFE_AEPTurbOPark_vs_AEPZong_s5_v2'
layout_p_grid.plot_diff_grid(dict_AEPTurbOPark['x_list'],dict_AEPTurbOPark['y_list'],dict_AEPZong['x_list'],dict_AEPZong['y_list'],sigma_D=sigma_D,savefig=savefig, pathfig=name_path, formatfig=name_format, namefig=namefig, include_xy_labels=False, figsize=figsize, halfrange=halfrange,cmap=cmap)

cmap = make_diverging_cmap(cmap_top=cmap_Jensen,cmap_bottom=cmap_Bastankhah)
dict_AEPJensen = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPJensen/data/',namefile_root='data_HKNscaled_AEPJensen_v',f_name_list=[])
dict_AEPBastankhah = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPBastankhah/data/',namefile_root='data_HKNscaled_AEPBastankhah_v',f_name_list=[])
namefig = 'LFE_AEPJensen_vs_AEPBastankhah_s5_v2'
layout_p_grid.plot_diff_grid(dict_AEPJensen['x_list'],dict_AEPJensen['y_list'],dict_AEPBastankhah['x_list'],dict_AEPBastankhah['y_list'],sigma_D=sigma_D,savefig=savefig, pathfig=name_path, formatfig=name_format, namefig=namefig, include_xy_labels=False, figsize=figsize, halfrange=halfrange,cmap=cmap)

cmap = make_diverging_cmap(cmap_top=cmap_Zong,cmap_bottom=cmap_Bastankhah)
dict_AEPZong = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPJensen/data/',namefile_root='data_HKNscaled_AEPJensen_v',f_name_list=[])
dict_AEPBastankhah = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPBastankhah/data/',namefile_root='data_HKNscaled_AEPBastankhah_v',f_name_list=[])
namefig = 'LFE_AEPZong_vs_AEPBastankhah_s5_v2'
layout_p_grid.plot_diff_grid(dict_AEPZong['x_list'],dict_AEPZong['y_list'],dict_AEPBastankhah['x_list'],dict_AEPBastankhah['y_list'],sigma_D=sigma_D,savefig=savefig, pathfig=name_path, formatfig=name_format, namefig=namefig, include_xy_labels=False, figsize=figsize, halfrange=halfrange,cmap=cmap)

cmap = make_diverging_cmap(cmap_top=cmap_Zong,cmap_bottom=cmap_Jensen)
dict_AEPZong = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPJensen/data/',namefile_root='data_HKNscaled_AEPJensen_v',f_name_list=[])
dict_AEPJensen = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPBastankhah/data/',namefile_root='data_HKNscaled_AEPBastankhah_v',f_name_list=[])
namefig = 'LFE_AEPZong_vs_AEPJensen_s5_v2'
layout_p_grid.plot_diff_grid(dict_AEPZong['x_list'],dict_AEPZong['y_list'],dict_AEPJensen['x_list'],dict_AEPJensen['y_list'],sigma_D=sigma_D,savefig=savefig, pathfig=name_path, formatfig=name_format, namefig=namefig, include_xy_labels=False, figsize=figsize, halfrange=halfrange,cmap=cmap)








#%%
# extract data - sensitivity analysis site characteristics


#%%
# extract data


dataname_list = [
    'AEPTurbOPark',
    'AEPJensen',
    'AEPBastankhah',
    'AEPZong',
    'AEPTurbOParkLsum',
    'AEPTurbOParkRcenter',
    'AEPTurbOParkCGI',
    'AEPTurbOPark5deg',
    'AEPTurbOPark10deg',
    'AEPTurbOPark8ms',
    'AEPTurbOParkaWR',
    'LCOETurbOPark',
    #'IOETurbOPark',
    'REVTurbOPark',
    'COVETurbOPark',
    'AEPTurbOParkGY',
    'AEPTurbOParkCab',
    'AEPTurbOParkSGD',
    'AEPTurbOParkLOGAf',
    ]

label_list = [
    'REF',
    'wdm-JENS',
    'wdm-BAST',
    'wdmZONG',
    'wsm-LSUM',
    'ram-RCEN',
    'ram-CGI',
    'wdd-5DEG',
    'wdd-10DEG',
    'wsd-1VAL',
    'wrd-HOM',
    'obj-LCOE',
    #'obj-IOE',
    'obj-REV',
    'obj-COVE',
    'cod-YAW',
    'cod-CAB',
    'opt-SGD',
    'opt-LOGAf',
]


category_list = [
    'Referent',
    'Wake deficit model',
    'Wake deficit model',
    'Wake deficit model',
    'Wake superposition model',
    'Rotor average model',
    'Rotor average model',
    'Wind direction discretization',
    'Wind direction discretization',
    'Wind speed discretization',
    'Wind resource data',
    'Objective function choice',
    #'Objective function choice',
    'Objective function choice',
    'Objective function choice',
    'Wind farm co-design',
    'Wind farm co-design',
    'Optimization algorithm',
    'Optimization algorithm',
    ]

category_list_unique = list(dict.fromkeys(category_list))



# extract LFIE - LPD
n_samples = np.array([1,2,3,4,5,6,7,8,9,10])
name_path = 'data_LF/'
root_name = 'HKNsLPD_'
data_LFIE_LPD = {}
for i in np.arange(len(dataname_list)):
    data_LFIE_LPD[dataname_list[i]] = {}
    with open(name_path+root_name+dataname_list[i]+'/'+root_name+dataname_list[i]+'_LFIE'+'.pkl', 'rb') as f:
        dict_LFIE = pickle.load(f)
    dict_LFIE_s = dict_LFIE['dict_LFIE_s']
    for j in np.arange(len(dataname_list)):
        data_LFIE_LPD[dataname_list[i]][dataname_list[j]] = dict_LFIE_s[root_name+dataname_list[j]]

# extract sigma array
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_LFIE'+'.pkl', 'rb') as f:
    dict_LFIE = pickle.load(f)
sigma_D_array = dict_LFIE['sigma_D_array']

# extract referent self comparison
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_self_LFIE'+'.pkl', 'rb') as f:
    dict_self_LFIE = pickle.load(f)
referent_self_lfie_ns_LPD = dict_self_LFIE['LFIE_ns']

# extract referent comparison
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_LFIE'+'.pkl', 'rb') as f:
    dict_LFIE = pickle.load(f)
dict_LFIE_s = dict_LFIE['dict_LFIE_s']
referent_lfie_s_LPD = dict_LFIE_s[root_name+'AEPTurbOPark2']



# extract LFIE - HPD
n_samples = np.array([1,2,3,4,5,6,7,8,9,10])
name_path = 'data_LF/'
root_name = 'HKNsHPD_'
data_LFIE_HPD = {}
for i in np.arange(len(dataname_list)):
    data_LFIE_HPD[dataname_list[i]] = {}
    with open(name_path+root_name+dataname_list[i]+'/'+root_name+dataname_list[i]+'_LFIE'+'.pkl', 'rb') as f:
        dict_LFIE = pickle.load(f)
    dict_LFIE_s = dict_LFIE['dict_LFIE_s']
    for j in np.arange(len(dataname_list)):
        data_LFIE_HPD[dataname_list[i]][dataname_list[j]] = dict_LFIE_s[root_name+dataname_list[j]]

# extract sigma array
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_LFIE'+'.pkl', 'rb') as f:
    dict_LFIE = pickle.load(f)
sigma_D_array = dict_LFIE['sigma_D_array']

# extract referent self comparison
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_self_LFIE'+'.pkl', 'rb') as f:
    dict_self_LFIE = pickle.load(f)
referent_self_lfie_ns_HPD = dict_self_LFIE['LFIE_ns']

# extract referent comparison
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_LFIE'+'.pkl', 'rb') as f:
    dict_LFIE = pickle.load(f)
dict_LFIE_s = dict_LFIE['dict_LFIE_s']
referent_lfie_s_HPD = dict_LFIE_s[root_name+'AEPTurbOPark2']



# extract LFIE - HBD
n_samples = np.array([1,2,3,4,5,6,7,8,9,10])
name_path = 'data_LF/'
root_name = 'HKNsHBD_'
data_LFIE_HBD = {}
for i in np.arange(len(dataname_list)):
    data_LFIE_HBD[dataname_list[i]] = {}
    with open(name_path+root_name+dataname_list[i]+'/'+root_name+dataname_list[i]+'_LFIE'+'.pkl', 'rb') as f:
        dict_LFIE = pickle.load(f)
    dict_LFIE_s = dict_LFIE['dict_LFIE_s']
    dict_LFIE_s ={("HKNsHBD_" + key[len("HKNscaled_"):]) if key.startswith("HKNscaled_") else key: value for key, value in dict_LFIE_s.items()}
    for j in np.arange(len(dataname_list)):
        data_LFIE_HBD[dataname_list[i]][dataname_list[j]] = dict_LFIE_s[root_name+dataname_list[j]]

# extract sigma array
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_LFIE'+'.pkl', 'rb') as f:
    dict_LFIE = pickle.load(f)
sigma_D_array = dict_LFIE['sigma_D_array']

# extract referent self comparison
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_self_LFIE'+'.pkl', 'rb') as f:
    dict_self_LFIE = pickle.load(f)
referent_self_lfie_ns_HBD = dict_self_LFIE['LFIE_ns']

# extract referent comparison
with open(name_path+root_name+dataname_list[0]+'/'+root_name+dataname_list[0]+'_LFIE'+'.pkl', 'rb') as f:
    dict_LFIE = pickle.load(f)
dict_LFIE_s = dict_LFIE['dict_LFIE_s']
dict_LFIE_s ={("HKNsHBD_" + key[len("HKNscaled_"):]) if key.startswith("HKNscaled_") else key: value for key, value in dict_LFIE_s.items()}
referent_lfie_s_HBD = dict_LFIE_s[root_name+'AEPTurbOPark2']



#%%
# sensitivity analysis: different case studies

# BL case study
ind_sigma = 2
lfie_row_BL = np.ones((len(dataname_list)))*np.nan
for i in np.arange(len(dataname_list)):
    lfie_row_BL[i] = data_LFIE[dataname_list[0]][dataname_list[i]][ind_sigma]
referent_self_d_eff = from_lfie_to_d_eff(np.median(referent_self_lfie_ns[:,ind_sigma],axis=(0)),sigma_D_array[ind_sigma])
referent_d_eff = from_lfie_to_d_eff(referent_lfie_s[ind_sigma],sigma_D_array[ind_sigma])
cmap_bounds_BL = [referent_d_eff,referent_self_d_eff]

# LPD case study
ind_sigma = 2
lfie_row_LPD = np.ones((len(dataname_list)))*np.nan
for i in np.arange(len(dataname_list)):
    lfie_row_LPD[i] = data_LFIE_LPD[dataname_list[0]][dataname_list[i]][ind_sigma]
referent_self_d_eff = from_lfie_to_d_eff(np.median(referent_self_lfie_ns_LPD[:,ind_sigma],axis=(0)),sigma_D_array[ind_sigma])
referent_d_eff = from_lfie_to_d_eff(referent_lfie_s_LPD[ind_sigma],sigma_D_array[ind_sigma])
cmap_bounds_LPD = [referent_d_eff,referent_self_d_eff]

# HPD case study
ind_sigma = 2
lfie_row_HPD = np.ones((len(dataname_list)))*np.nan
for i in np.arange(len(dataname_list)):
    lfie_row_HPD[i] = data_LFIE_HPD[dataname_list[0]][dataname_list[i]][ind_sigma]
referent_self_d_eff = from_lfie_to_d_eff(np.median(referent_self_lfie_ns_HPD[:,ind_sigma],axis=(0)),sigma_D_array[ind_sigma])
referent_d_eff = from_lfie_to_d_eff(referent_lfie_s_HPD[ind_sigma],sigma_D_array[ind_sigma])
cmap_bounds_HPD = [referent_d_eff,referent_self_d_eff]

# HBD case study
ind_sigma = 2
lfie_row_HBD = np.ones((len(dataname_list)))*np.nan
for i in np.arange(len(dataname_list)):
    lfie_row_HBD[i] = data_LFIE_HBD[dataname_list[0]][dataname_list[i]][ind_sigma]
referent_self_d_eff = from_lfie_to_d_eff(np.median(referent_self_lfie_ns_HBD[:,ind_sigma],axis=(0)),sigma_D_array[ind_sigma])
referent_d_eff = from_lfie_to_d_eff(referent_lfie_s_HBD[ind_sigma],sigma_D_array[ind_sigma])
cmap_bounds_HBD = [referent_d_eff,referent_self_d_eff]


# concat matrices
lfie_mat_sensAn = np.concatenate((lfie_row_BL[na,:],lfie_row_LPD[na,:],lfie_row_HPD[na,:],lfie_row_HBD[na,:]),axis=(0))
cmap_bounds_list = [cmap_bounds_BL,cmap_bounds_LPD,cmap_bounds_HPD,cmap_bounds_HBD]


savefig = False
name_path = '../figures'
name_format = 'svg'


# d eff mat (v4)
fig, ax = plt.subplots(figsize=(8.5,2))
plot_matrix_lfie_sensAn_v4(from_lfie_to_d_eff(lfie_mat_sensAn,sigma_D_array[ind_sigma]),fig,ax,cases_list=label_list,rowname_list=['Baseline','Low power density','High power density','High bathymetry diff.'],cmap_bounds_list=cmap_bounds_list,color=plt.get_cmap('Blues')(0.95),sizes=[0.2,0.4,0.85])
if savefig: plt.savefig(name_path+'\\'+'deff_mat_sensAn'+'.'+name_format,format=name_format,bbox_inches='tight')
plt.show()

# %%
