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


#%%
# functions

# normalized difference between median values
def ndf(f,f0):
    return 100*(np.median(f)-np.median(f0))/np.abs(np.median (f0))

# normalized difference between median values - array
def ndf_array(f,f0):
    return 100*(f-np.median(f0))/np.abs(np.median(f0))

# create ndf fval matrix (allows to specify the normalization term for each element of f_name_list)
def create_fval_diff_mat_v2(dict_list,f_name_list,dict_norm_list):
    fval_diff_mat = np.ones((len(f_name_list),len(dict_list)))*np.nan
    for i in np.arange(len(f_name_list)):
        for j in np.arange(len(dict_list)):
            fval_diff_mat[i,j] = ndf(dict_list[j][f_name_list[i]],dict_norm_list[i][f_name_list[i]])
    return fval_diff_mat

# create ndf fval array list (one evaluation function)
def create_fval_diff_mat_list(dict_list,dict_ref,f_name):
    fval_diff_list = [None]*len(dict_list)
    for j in np.arange(len(dict_list)):
        fval_diff_list[j] = ndf_array(dict_list[j][f_name],dict_ref[f_name])
    return fval_diff_list

# create filter matrix
def create_category_filter(category_list_obj,category_list_eval):
    mat_filter = np.zeros((len(category_list_eval),len(category_list_obj)),dtype=bool)
    for i in np.arange(len(category_list_eval)):
        for j in np.arange(len(category_list_obj)):
            mat_filter[i,j] = category_list_eval[i]==category_list_obj[j]
    mat_filter[category_list_eval.index('Referent'),:] = True
    mat_filter[:,category_list_obj.index('Referent')] = True
    return mat_filter



def plot_matrix(mat,
                fig,
                ax,
                mat_filter=None,
                cmap='RdYlGn', # list of len(cmap_bouds)+1 elements 
                title='',
                cases_list=None,
                pad=10,
                direct_comparison_mat=None):
    
    bound_val = np.maximum(-np.nanmin(mat),np.nanmax(mat))
    norm = mcolors.TwoSlopeNorm(vmin=-bound_val, vcenter=0, vmax=bound_val)
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
    ax.set_ylabel('Evaluation choice')
    ax.tick_params(axis='x', pad=pad)
    ax.tick_params(axis='y', pad=pad)
    plt.setp(ax.get_xticklabels(), ha="right", va="top",rotation=60)
    plt.setp(ax.get_yticklabels(), ha="right", va="center", rotation=0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            s = 0.75
            x = j
            y = mat.shape[0] - 1 - i
            rect = plt.Rectangle((x-s/2,y-s/2),s,s,facecolor=cmap(norm(mat[i,j])),edgecolor='white',linewidth=3)
            ax.add_patch(rect)
            if ~direct_comparison_mat[i,j]:
                rect = plt.Rectangle((x-0.5,y-0.5),1,1,fill=False,hatch='/////////',edgecolor='white',linewidth=5)
                ax.add_patch(rect)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    #sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Difference in the evaluation function [%]')

    scale = 1+15*s
    hatch_legend = [patches.Rectangle((0, 0),scale,scale,facecolor='gray',edgecolor='white',label='Direct'),patches.Rectangle((0, 0),scale,scale,facecolor='gray',edgecolor='white',hatch='/////////',label='Indirect')]
    leg2 = ax.legend(handles=hatch_legend,title='Comparison',loc='lower center',bbox_to_anchor=(0.85, 1.02),ncol=2,handlelength=1.0,handleheight=1.0)


def plot_matrix_sensAn(mat,
                       fig,
                       ax,
                       mat_filter=None,
                       cmap='RdYlGn', # list of len(cmap_bouds)+1 elements 
                       title='',
                       cases_list=None,
                       case_study_list=None,
                       pad=10):
    
    bound_val = np.maximum(-np.nanmin(mat),np.nanmax(mat))
    norm = mcolors.TwoSlopeNorm(vmin=-bound_val, vcenter=0, vmax=bound_val)
    cmap = plt.get_cmap(cmap)
     
    if mat_filter is not None:
        mat[~mat_filter] = np.nan

    ax.set_title(title)
    ax.set_xlim(-0.5,mat.shape[1]-0.5)
    ax.set_ylim(-0.5,mat.shape[0]-0.5)
    ax.set_xticks(range(len(cases_list)))
    ax.set_yticks(np.flip(range(len(case_study_list))))
    ax.set_xticklabels(cases_list)
    ax.set_yticklabels(case_study_list)
    ax.set_xlabel('Design choice')
    #ax.set_ylabel('Case study')
    ax.tick_params(axis='x', pad=pad)
    ax.tick_params(axis='y', pad=pad)
    plt.setp(ax.get_xticklabels(), ha="right", va="top",rotation=60)
    plt.setp(ax.get_yticklabels(), ha="right", va="center", rotation=0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            s = 0.75
            x = j
            y = mat.shape[0] - 1 - i
            rect = plt.Rectangle((x-s/2,y-s/2),s,s,facecolor=cmap(norm(mat[i,j])),edgecolor='white',linewidth=3)
            ax.add_patch(rect)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Difference in the\nevaluation function\n[%]')






# function to create dictionary
def create_dict_case(n_samples,name_path,namefile_root,f_name_list,invert_sign_list=[],fix_REVTurbOPark=False):
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
                if f_name in invert_sign_list:
                    dict_case[f_name][i] = -1*data.get(f_name, data.get(f_name + '()'))#data[f_name]    
                else:
                    dict_case[f_name][i] = data.get(f_name, data.get(f_name + '()'))#data[f_name]    
    if fix_REVTurbOPark:
        dict_case['REVTurbOPark'] = dict_case['REVTurbOPark']*(8760*1e-9)
    return dict_case


#%%
# extract data

f_name_list = [
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
    'REVTurbOPark',
    'COVETurbOPark',
    'AEPTurbOParkGY',
    'AEPTurbOParkCab',
    'AEPTurbOPark',
    'AEPTurbOPark',
    ]

n_samples = np.array([1,2,3,4,5,6,7,8,9,10])


dict_AEPTurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOPark/data/',namefile_root='data_HKNscaled_AEPTurbOPark_v',f_name_list=f_name_list)
dict_AEPJensen = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPJensen/data/',namefile_root='data_HKNscaled_AEPJensen_v',f_name_list=f_name_list)
dict_AEPBastankhah = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPBastankhah/data/',namefile_root='data_HKNscaled_AEPBastankhah_v',f_name_list=f_name_list)
dict_AEPZong = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPZong/data/',namefile_root='data_HKNscaled_AEPZong_v',f_name_list=f_name_list)
dict_AEPTurbOParkLsum = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkLsum/data/',namefile_root='data_HKNscaled_AEPTurbOParkLsum_v',f_name_list=f_name_list)
dict_AEPTurbOParkRcenter = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkRcenter/data/',namefile_root='data_HKNscaled_AEPTurbOParkRcenter_v',f_name_list=f_name_list)
dict_AEPTurbOParkCGI = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkCGI/data/',namefile_root='data_HKNscaled_AEPTurbOParkCGI_v',f_name_list=f_name_list)
dict_AEPTurbOPark5deg = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOPark5deg/data/',namefile_root='data_HKNscaled_AEPTurbOPark5deg_v',f_name_list=f_name_list)
dict_AEPTurbOPark10deg = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOPark10deg/data/',namefile_root='data_HKNscaled_AEPTurbOPark10deg_v',f_name_list=f_name_list)
dict_AEPTurbOPark8ms = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOPark8ms/data/',namefile_root='data_HKNscaled_AEPTurbOPark8ms_v',f_name_list=f_name_list)
dict_AEPTurbOParkaWR = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkaWR/data/',namefile_root='data_HKNscaled_AEPTurbOParkaWR_v',f_name_list=f_name_list)
dict_LCOETurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_LCOETurbOPark/data/',namefile_root='data_HKNscaled_LCOETurbOPark_v',f_name_list=f_name_list)
dict_REVTurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_REVTurbOPark/data/',namefile_root='data_HKNscaled_REVTurbOPark_v',f_name_list=f_name_list)
dict_COVETurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_COVETurbOPark/data/',namefile_root='data_HKNscaled_COVETurbOPark_v',f_name_list=f_name_list)
dict_AEPTurbOParkGY = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkGY/data/',namefile_root='data_HKNscaled_AEPTurbOParkGY_v',f_name_list=f_name_list)
dict_AEPTurbOParkCab = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkCab/data/',namefile_root='data_HKNscaled_AEPTurbOParkCab_v',f_name_list=f_name_list)
dict_AEPTurbOParkSGD = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkSGD/data/',namefile_root='data_HKNscaled_AEPTurbOParkSGD_v',f_name_list=f_name_list)
dict_AEPTurbOParkLOGAf = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkLOGAf/data/',namefile_root='data_HKNscaled_AEPTurbOParkLOGAf_v',f_name_list=f_name_list)


dict_list = [
    dict_AEPTurbOPark,
    dict_AEPJensen,
    dict_AEPBastankhah,
    dict_AEPZong,
    dict_AEPTurbOParkLsum,
    dict_AEPTurbOParkRcenter,
    dict_AEPTurbOParkCGI,
    dict_AEPTurbOPark5deg,
    dict_AEPTurbOPark10deg,
    dict_AEPTurbOPark8ms,
    dict_AEPTurbOParkaWR,
    dict_LCOETurbOPark,
    dict_REVTurbOPark,
    dict_COVETurbOPark,
    dict_AEPTurbOParkGY,
    dict_AEPTurbOParkCab,
    dict_AEPTurbOParkSGD,
    dict_AEPTurbOParkLOGAf,
]

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
    'Objective function choice',
    'Objective function choice',
    'Wind farm co-design',
    'Wind farm co-design',
    'Optimization algorithm',
    'Optimization algorithm',
    ]

category_list_unique = list(dict.fromkeys(category_list))


#%%
# plot boxplot (only negative impact of using another function instead of the REFERENT)

savefig = False
name_path = '../figures'
name_format = 'svg'

# initialize
fval_diff_list_boxplot = [None]*len(dict_list)

# calculate fval_diff for each evaluation case
for i in np.arange(len(dict_list)):
    fval_diff_list_boxplot[i] = create_fval_diff_mat_list(dict_list=[dict_list[i]],
                                            dict_ref=dict_AEPTurbOPark,
                                            f_name='AEPTurbOPark')

colors = ['#000000', "#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#bbbbbb", "#ffa500", "#44aa99" ]
colors_bp = [colors[0]]*1+[colors[1]]*3+[colors[2]]*1+[colors[3]]*2+[colors[4]]*2+[colors[5]]*1+[colors[6]]*1+[colors[7]]*3+[colors[8]]*2+[colors[9]]*2

fig,ax = plt.subplots(figsize=(8,5))

# draw horizontal line
x_plot = np.arange(len(dict_list))
plt.plot([np.min(x_plot)-0.5,np.max(x_plot)+0.5],[0,0],c='k')

# create boxplot
for i in np.arange(len(dict_list)):
    flierprops = dict(marker='o',markersize=3,markerfacecolor=colors_bp[i],markeredgecolor='black',linestyle='none')
    bp = ax.boxplot(fval_diff_list_boxplot[i],positions=[x_plot[i]],widths=0.5,patch_artist=True,showfliers=True,medianprops=dict(color='black'),flierprops=flierprops)
    bp['boxes'][0].set_facecolor(colors_bp[i])

# set labels
ax.set_xticks(x_plot)
ax.set_xticklabels(label_list,rotation=45,ha='right')
ax.set_xlabel('Design choice')
ax.set_ylabel('Difference in evaluation function [%]')

handles = [ ax.scatter([],[],marker='s',color=colors[i],label=category_list_unique[i],edgecolor='k') for i in np.arange(len(category_list_unique))]
legend = ax.legend(handles=handles,ncols=2,loc='lower center', bbox_to_anchor=(0.5,1.02))

plt.tight_layout()
if savefig: plt.savefig(name_path+'\\'+'fval_boxplot'+'.'+name_format,format=name_format,bbox_inches='tight')
plt.show()




# %%
# plot matrix

savefig = False
name_path = '../figures'
name_format = 'svg'


# normalization: evaluation=design
mat_filter = create_category_filter(category_list,category_list)
fval_mat = create_fval_diff_mat_v2(dict_list=dict_list,f_name_list=f_name_list,dict_norm_list=dict_list)
mask = np.isin(category_list,['Wind direction discretization','Wind speed discretization','Wind resource data','Optimization algorithm'])
fval_mat = fval_mat.copy()
fval_mat[mask,:] = np.nan
fig, ax = plt.subplots(figsize=(10,7))
plot_matrix(fval_mat,fig,ax,mat_filter=None,cmap='PiYG',title='',cases_list=label_list,pad=10,direct_comparison_mat=mat_filter)
if savefig: plt.savefig(name_path+'\\'+'fval_mat_diagnorm'+'.'+name_format,format=name_format,bbox_inches='tight')
plt.show()




#%%
# extract data: sensitivity analysis


f_name_list = [
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
    'REVTurbOPark',
    'COVETurbOPark',
    'AEPTurbOParkGY',
    'AEPTurbOParkCab',
    'AEPTurbOPark',
    'AEPTurbOPark',
    ]

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
    'Objective function choice',
    'Objective function choice',
    'Wind farm co-design',
    'Wind farm co-design',
    'Optimization algorithm',
    'Optimization algorithm',
    ]

category_list_unique = list(dict.fromkeys(category_list))

n_samples = np.array([1,2,3,4,5,6,7,8,9,10])


# LPD
dict_AEPTurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOPark/data/',namefile_root='data_HKNsLPD_AEPTurbOPark_v',f_name_list=f_name_list)
dict_AEPJensen = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPJensen/data/',namefile_root='data_HKNsLPD_AEPJensen_v',f_name_list=f_name_list)
dict_AEPBastankhah = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPBastankhah/data/',namefile_root='data_HKNsLPD_AEPBastankhah_v',f_name_list=f_name_list)
dict_AEPZong = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPZong/data/',namefile_root='data_HKNsLPD_AEPZong_v',f_name_list=f_name_list)
dict_AEPTurbOParkLsum = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOParkLsum/data/',namefile_root='data_HKNsLPD_AEPTurbOParkLsum_v',f_name_list=f_name_list)
dict_AEPTurbOParkRcenter = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOParkRcenter/data/',namefile_root='data_HKNsLPD_AEPTurbOParkRcenter_v',f_name_list=f_name_list)
dict_AEPTurbOParkCGI = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOParkCGI_v2/data/',namefile_root='data_HKNsLPD_AEPTurbOParkCGI_v',f_name_list=f_name_list)
dict_AEPTurbOPark5deg = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOPark5deg/data/',namefile_root='data_HKNsLPD_AEPTurbOPark5deg_v',f_name_list=f_name_list)
dict_AEPTurbOPark10deg = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOPark10deg/data/',namefile_root='data_HKNsLPD_AEPTurbOPark10deg_v',f_name_list=f_name_list)
dict_AEPTurbOPark8ms = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOPark8ms/data/',namefile_root='data_HKNsLPD_AEPTurbOPark8ms_v',f_name_list=f_name_list)
dict_AEPTurbOParkaWR = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOParkaWR/data/',namefile_root='data_HKNsLPD_AEPTurbOParkaWR_v',f_name_list=f_name_list)
dict_LCOETurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_LCOETurbOPark/data/',namefile_root='data_HKNsLPD_LCOETurbOPark_v',f_name_list=f_name_list)
dict_REVTurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_REVTurbOPark/data/',namefile_root='data_HKNsLPD_REVTurbOPark_v',f_name_list=f_name_list)
dict_COVETurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_COVETurbOPark/data/',namefile_root='data_HKNsLPD_COVETurbOPark_v',f_name_list=f_name_list)
dict_AEPTurbOParkGY = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOParkGY_v2/data/',namefile_root='data_HKNsLPD_AEPTurbOParkGY_v',f_name_list=f_name_list)
dict_AEPTurbOParkCab = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOParkCab/data/',namefile_root='data_HKNsLPD_AEPTurbOParkCab_v',f_name_list=f_name_list)
dict_AEPTurbOParkSGD = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOParkSGD/data/',namefile_root='data_HKNsLPD_AEPTurbOParkSGD_v',f_name_list=f_name_list)
dict_AEPTurbOParkLOGAf = create_dict_case(n_samples,name_path='data_fobj/HKNscaledLPD/HKNsLPD_AEPTurbOParkLOGAf/data/',namefile_root='data_HKNsLPD_AEPTurbOParkLOGAf_v',f_name_list=f_name_list)

dict_list_LPD = [
    dict_AEPTurbOPark,
    dict_AEPJensen,
    dict_AEPBastankhah,
    dict_AEPZong,
    dict_AEPTurbOParkLsum,
    dict_AEPTurbOParkRcenter,
    dict_AEPTurbOParkCGI,
    dict_AEPTurbOPark5deg,
    dict_AEPTurbOPark10deg,
    dict_AEPTurbOPark8ms,
    dict_AEPTurbOParkaWR,
    dict_LCOETurbOPark,
    dict_REVTurbOPark,
    dict_COVETurbOPark,
    dict_AEPTurbOParkGY,
    dict_AEPTurbOParkCab,
    dict_AEPTurbOParkSGD,
    dict_AEPTurbOParkLOGAf,
]


# HPD
dict_AEPTurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOPark/data/',namefile_root='data_HKNsHPD_AEPTurbOPark_v',f_name_list=f_name_list)
dict_AEPJensen = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPJensen/data/',namefile_root='data_HKNsHPD_AEPJensen_v',f_name_list=f_name_list)
dict_AEPBastankhah = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPBastankhah/data/',namefile_root='data_HKNsHPD_AEPBastankhah_v',f_name_list=f_name_list)
dict_AEPZong = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPZong/data/',namefile_root='data_HKNsHPD_AEPZong_v',f_name_list=f_name_list)
dict_AEPTurbOParkLsum = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOParkLsum/data/',namefile_root='data_HKNsHPD_AEPTurbOParkLsum_v',f_name_list=f_name_list)
dict_AEPTurbOParkRcenter = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOParkRcenter/data/',namefile_root='data_HKNsHPD_AEPTurbOParkRcenter_v',f_name_list=f_name_list)
dict_AEPTurbOParkCGI = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOParkCGI_v2/data/',namefile_root='data_HKNsHPD_AEPTurbOParkCGI_v',f_name_list=f_name_list)
dict_AEPTurbOPark5deg = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOPark5deg/data/',namefile_root='data_HKNsHPD_AEPTurbOPark5deg_v',f_name_list=f_name_list)
dict_AEPTurbOPark10deg = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOPark10deg/data/',namefile_root='data_HKNsHPD_AEPTurbOPark10deg_v',f_name_list=f_name_list)
dict_AEPTurbOPark8ms = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOPark8ms/data/',namefile_root='data_HKNsHPD_AEPTurbOPark8ms_v',f_name_list=f_name_list)
dict_AEPTurbOParkaWR = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOParkaWR/data/',namefile_root='data_HKNsHPD_AEPTurbOParkaWR_v',f_name_list=f_name_list)
dict_LCOETurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_LCOETurbOPark/data/',namefile_root='data_HKNsHPD_LCOETurbOPark_v',f_name_list=f_name_list)
dict_REVTurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_REVTurbOPark/data/',namefile_root='data_HKNsHPD_REVTurbOPark_v',f_name_list=f_name_list)
dict_COVETurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_COVETurbOPark/data/',namefile_root='data_HKNsHPD_COVETurbOPark_v',f_name_list=f_name_list)
dict_AEPTurbOParkGY = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOParkGY_v2/data/',namefile_root='data_HKNsHPD_AEPTurbOParkGY_v',f_name_list=f_name_list)
dict_AEPTurbOParkCab = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOParkCab/data/',namefile_root='data_HKNsHPD_AEPTurbOParkCab_v',f_name_list=f_name_list)
dict_AEPTurbOParkSGD = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOParkSGD/data/',namefile_root='data_HKNsHPD_AEPTurbOParkSGD_v',f_name_list=f_name_list)
dict_AEPTurbOParkLOGAf = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHPD/HKNsHPD_AEPTurbOParkLOGAf/data/',namefile_root='data_HKNsHPD_AEPTurbOParkLOGAf_v',f_name_list=f_name_list)

dict_list_HPD = [
    dict_AEPTurbOPark,
    dict_AEPJensen,
    dict_AEPBastankhah,
    dict_AEPZong,
    dict_AEPTurbOParkLsum,
    dict_AEPTurbOParkRcenter,
    dict_AEPTurbOParkCGI,
    dict_AEPTurbOPark5deg,
    dict_AEPTurbOPark10deg,
    dict_AEPTurbOPark8ms,
    dict_AEPTurbOParkaWR,
    dict_LCOETurbOPark,
    dict_REVTurbOPark,
    dict_COVETurbOPark,
    dict_AEPTurbOParkGY,
    dict_AEPTurbOParkCab,
    dict_AEPTurbOParkSGD,
    dict_AEPTurbOParkLOGAf,
]


# HBD
dict_AEPTurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOPark/data/',namefile_root='data_HKNscaled_AEPTurbOPark_v',f_name_list=f_name_list)
dict_AEPJensen = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPJensen/data/',namefile_root='data_HKNscaled_AEPJensen_v',f_name_list=f_name_list)
dict_AEPBastankhah = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPBastankhah/data/',namefile_root='data_HKNscaled_AEPBastankhah_v',f_name_list=f_name_list)
dict_AEPZong = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPZong/data/',namefile_root='data_HKNscaled_AEPZong_v',f_name_list=f_name_list)
dict_AEPTurbOParkLsum = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkLsum/data/',namefile_root='data_HKNscaled_AEPTurbOParkLsum_v',f_name_list=f_name_list)
dict_AEPTurbOParkRcenter = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkRcenter/data/',namefile_root='data_HKNscaled_AEPTurbOParkRcenter_v',f_name_list=f_name_list)
dict_AEPTurbOParkCGI = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkCGI/data/',namefile_root='data_HKNscaled_AEPTurbOParkCGI_v',f_name_list=f_name_list)
dict_AEPTurbOPark5deg = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOPark5deg/data/',namefile_root='data_HKNscaled_AEPTurbOPark5deg_v',f_name_list=f_name_list)
dict_AEPTurbOPark10deg = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOPark10deg/data/',namefile_root='data_HKNscaled_AEPTurbOPark10deg_v',f_name_list=f_name_list)
dict_AEPTurbOPark8ms = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOPark8ms/data/',namefile_root='data_HKNscaled_AEPTurbOPark8ms_v',f_name_list=f_name_list)
dict_AEPTurbOParkaWR = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkaWR/data/',namefile_root='data_HKNscaled_AEPTurbOParkaWR_v',f_name_list=f_name_list)
dict_LCOETurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHBD/HKNsHBD_LCOETurbOPark/data/',namefile_root='data_HKNsHBD_LCOETurbOPark_v',f_name_list=f_name_list)
dict_REVTurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_REVTurbOPark/data/',namefile_root='data_HKNscaled_REVTurbOPark_v',f_name_list=f_name_list)
dict_COVETurbOPark = create_dict_case(n_samples,name_path='data_fobj/HKNscaledHBD/HKNsHBD_COVETurbOPark/data/',namefile_root='data_HKNsHBD_COVETurbOPark_v',f_name_list=f_name_list)
dict_AEPTurbOParkGY = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkGY/data/',namefile_root='data_HKNscaled_AEPTurbOParkGY_v',f_name_list=f_name_list)
dict_AEPTurbOParkCab = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkCab/data/',namefile_root='data_HKNscaled_AEPTurbOParkCab_v',f_name_list=f_name_list)
dict_AEPTurbOParkSGD = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkSGD/data/',namefile_root='data_HKNscaled_AEPTurbOParkSGD_v',f_name_list=f_name_list)
dict_AEPTurbOParkLOGAf = create_dict_case(n_samples,name_path='data_fobj/HKNscaled/HKNscaled_AEPTurbOParkLOGAf/data/',namefile_root='data_HKNscaled_AEPTurbOParkLOGAf_v',f_name_list=f_name_list)

dict_list_HBD = [
    dict_AEPTurbOPark,
    dict_AEPJensen,
    dict_AEPBastankhah,
    dict_AEPZong,
    dict_AEPTurbOParkLsum,
    dict_AEPTurbOParkRcenter,
    dict_AEPTurbOParkCGI,
    dict_AEPTurbOPark5deg,
    dict_AEPTurbOPark10deg,
    dict_AEPTurbOPark8ms,
    dict_AEPTurbOParkaWR,
    dict_LCOETurbOPark,
    #dict_IOETurbOPark,
    dict_REVTurbOPark,
    dict_COVETurbOPark,
    dict_AEPTurbOParkGY,
    dict_AEPTurbOParkCab,
    dict_AEPTurbOParkSGD,
    dict_AEPTurbOParkLOGAf,
]





#%%
# plot matrix: sensitivity analysis

savefig = False
name_path = '../figures'
name_format = 'svg'


# normalization: evaluation=design

case_study_list = ['Baseline','Low power density','High power density','High bathymetry diff.']
fval_row_BL = create_fval_diff_mat_v2(dict_list=dict_list,f_name_list=[f_name_list[0]],dict_norm_list=[dict_list[0]])
fval_row_LPD = create_fval_diff_mat_v2(dict_list=dict_list_LPD,f_name_list=[f_name_list[0]],dict_norm_list=[dict_list_LPD[0]])
fval_row_HPD = create_fval_diff_mat_v2(dict_list=dict_list_HPD,f_name_list=[f_name_list[0]],dict_norm_list=[dict_list_HPD[0]])
fval_row_HBD = create_fval_diff_mat_v2(dict_list=dict_list_HBD,f_name_list=[f_name_list[0]],dict_norm_list=[dict_list_HBD[0]])

fval_mat_sensAn = np.concatenate((fval_row_BL,fval_row_LPD,fval_row_HPD,fval_row_HBD),axis=(0))

fig, ax = plt.subplots(figsize=(10,1.7))
plot_matrix_sensAn(fval_mat_sensAn,fig,ax,mat_filter=None,cmap='PiYG',title='',cases_list=label_list,case_study_list=case_study_list,pad=10)
if savefig: plt.savefig(name_path+'\\'+'fval_mat_diagnorm_sensAn'+'.'+name_format,format=name_format,bbox_inches='tight')
plt.show()





# %%
