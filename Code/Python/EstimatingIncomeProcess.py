# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: collapsed,code_folding
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.6
#   toc:
#     base_numbering: 1
#     nav_menu: {}
#     number_sections: true
#     sideBar: true
#     skip_h1_title: false
#     title_cell: Table of Contents
#     title_sidebar: Contents
#     toc_cell: false
#     toc_position: {}
#     toc_section_display: true
#     toc_window_display: false
# ---

# %% [markdown]
#
# # Estimating Income Processes
#
# The permanent/transitory decomposition has held up well, but some questions remain. Scandinavian data provides us an unparalled opportunitity to measure this in more detail

# %% {"code_folding": [0]}
# Initial imports and notebook setup, click arrow to show
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ipywidgets as widgets
from min_distance import parameter_estimation, parameter_estimation_by_subgroup, vech_indices, implied_inc_cov_composite


# %% {"code_folding": []}
# Load empirical moments and variance matrix
moments_BPP_dir = Path("../../Data/BPP_moments/") 
empirical_moments_all = np.genfromtxt(Path(moments_BPP_dir,"moments_all_c_vector.txt"), delimiter=',')
Omega_all =    np.genfromtxt(Path(moments_BPP_dir,"moments_all_omega.txt"), delimiter=',')
moments_CS_dir = Path("../../Data/CarrollSamwickMoments/") 
empirical_moments_CS_all = np.genfromtxt(Path(moments_CS_dir,"moments_all_c_vector.csv"), delimiter=',')
T=12
#Just doing income for now - remove other moments
income_moments = np.array([[False]*2*T]*2*T, dtype=bool)
income_moments[T:,T:] = True
vech_indices2T = vech_indices(2*T)
income_moments = income_moments[vech_indices2T]
empirical_moments_inc = empirical_moments_all[income_moments]
Omega_inc = Omega_all[income_moments,:][:,income_moments]


# %% {"code_folding": [0]}
# set up estimation: initial guess, which parameters to estimate, and bounds
init_params = np.array([0.00809118,  #permanent variance
                        0.00355567,  #transitory variance
                        1.13118306,    #decay parameter of slightly persistant transitory shock
                        0.57210194,   #fraction of transitory variance that has no persistence
                        0.0])  # decay parameter of perm shock
optimize_index = np.array([True,  #permanent variance
                        True,  #transitory variance
                        True,    #decay parameter of slightly persistant transitory shock
                        True,   #fraction of transitory variance that has no persistence
                        False]) # decay parameter of perm shock (turned off for now)
bounds     = [(0.000001,0.1),
              (0.000001,0.1),
              (0.01,5.0),
              (0.0,1.0),
              (-0.1,0.1)]


# %% {"code_folding": []}
# Estimate parameters and calculate mean of moments over years
estimates, estimate_se = parameter_estimation(empirical_moments_inc, Omega_inc, T, init_params, bounds=bounds, optimize_index=optimize_index)  
implied_cov_full = implied_inc_cov_composite(estimates,T)
implied_cov = implied_cov_full[0:T]
# Get Carroll Samwick moments (just appropriate sum of BPP moments)
vech_indicesT = vech_indices(T)
def CS_from_BPP(BPP_moments):
    CS_moments = np.zeros((T,T))
    BPP_moments_matrix = np.zeros((T,T))
    BPP_moments_matrix[vech_indicesT] = BPP_moments
    BPP_moments_matrix[(vech_indicesT[1],vech_indicesT[0])] = BPP_moments
    for j in range(T):
        for i in np.array(range(T-j))+j:
            CS_moments[i,j] = np.sum(BPP_moments_matrix[j:i+1,j:i+1])
    CS_moments = CS_moments[vech_indicesT]
    return CS_moments
CS_moments = CS_from_BPP(empirical_moments_inc)
implied_CS_moments = CS_from_BPP(implied_cov_full)[0:T]
# Calculate mean empirical moments and standard errors
mean_moments = np.zeros(T)
mean_moments_se = np.zeros(T)
CS_moments_mean = np.zeros(T)
for t in range(T):
    this_diag = np.diag(1.0/(T-t)*np.ones(T-t),-t)
    this_diag = this_diag[vech_indicesT]
    mean_moments[t] = np.dot(this_diag,empirical_moments_inc)
    mean_moments_se[t] = np.dot(np.dot(this_diag,Omega_inc),this_diag)**0.5
    CS_moments_mean[t] = np.dot(this_diag,CS_moments)


# %% {"code_folding": []}
# Define plotting function

def compare_to_moments(compare, quantile):
    '''
    Gets the relevant empirical data for comparison graph
    '''
    if (compare=="All Households"):
        compare_moments_inc = empirical_moments_inc
        compare_Omega_inc = Omega_inc
        quantile_widget.options=['1']
    else:
        if (compare=="Liquid Wealth (quintiles)"):
            subgroup_stub='moments_by_liquid_wealth_quantile'
            quantile_widget.options=['1','2','3','4','5']
        elif (compare=="Net Wealth (quintiles)"):
            subgroup_stub='moments_by_net_wealth_quantile'
            quantile_widget.options=['1','2','3','4','5']
        elif (compare=="Income (deciles)"):
            subgroup_stub='moments_by_Income_quantile'
            quantile_widget.options=['1','2','3','4','5','6','7','8','9','10']
        elif (compare=="Net Nominal Position (deciles)"):
            subgroup_stub='moments_by_NNP_quantile'
            quantile_widget.options=['1','2','3','4','5','6','7','8','9','10']
        elif (compare=="Interest Rate Exposure (deciles)"):
            subgroup_stub='moments_by_URE_quantile'
            quantile_widget.options=['1','2','3','4','5','6','7','8','9','10']
        elif (compare=="Consumption (deciles)"):
            subgroup_stub='moments_by_MeanCons_quantile'
            quantile_widget.options=['1','2','3','4','5','6','7','8','9','10']
        compare_inc_name = subgroup_stub+str(quantile)+"c_vector.txt"
        compare_moments_all = np.genfromtxt(Path(moments_BPP_dir,compare_inc_name), delimiter=',')
        compare_moments_inc = compare_moments_all[income_moments]
        compare_Omega_name = subgroup_stub+str(quantile)+"_Omega.txt"
        compare_Omega_all =    np.genfromtxt(Path(moments_BPP_dir,compare_Omega_name), delimiter=',')
        compare_Omega_inc = Omega_all[income_moments,:][:,income_moments]
    return compare_moments_inc, compare_Omega_inc

def plot_moments(perm_var,tran_var,half_life,bonus,perm_decay,compare="All Households",quantile=1):
    fig = plt.figure(figsize=(14, 9),constrained_layout=True)
    gs = fig.add_gridspec(2, 13)
    panel1 = fig.add_subplot(gs[0, 0:3])
    panel2 = fig.add_subplot(gs[0, 4:])
    #panel3 = fig.add_subplot(gs[1, 0:3])
    panel4 = fig.add_subplot(gs[1, 1:-2])
    
    panel1.plot(mean_moments[0:3], marker='o')
    panel2.plot(mean_moments, marker='o',label="All Households Mean")
    CS_Ndiff = np.array(range(T))+1.0
    CS_moments_factor = (CS_Ndiff-1.0/3.0)
    #panel3.plot(CS_Ndiff[0:3],CS_moments_mean[0:3]/CS_moments_factor[0:3], marker='o')
    panel4.plot(CS_Ndiff,CS_moments_mean/CS_moments_factor, marker='o',label="All Households Mean")
    # Standard errors
    panel1.plot(mean_moments[0:3]+1.96*mean_moments_se[0:3],linestyle="--",color="gray",linewidth=1.0)
    panel1.plot(mean_moments[0:3]-1.96*mean_moments_se[0:3],linestyle="--",color="gray",linewidth=1.0)
    panel2.plot(mean_moments+1.96*mean_moments_se,linestyle="--",color="gray",linewidth=1.0)
    panel2.plot(mean_moments-1.96*mean_moments_se,linestyle="--",color="gray",linewidth=1.0)
    
    #plot the moments for each year
    panel1.plot(empirical_moments_inc[0:3], marker='x',linewidth=0,color='#1f77b4')
    panel2.plot(np.array(range(T)),empirical_moments_inc[0:T], marker='x',linewidth=0,label="Individual years",color='#1f77b4')
    #panel3.plot(CS_Ndiff[0:3],CS_moments[0:3]/CS_moments_factor[0:3], marker='x',linewidth=0,color='#1f77b4')
    panel4.plot(CS_Ndiff[0:T],CS_moments[0:T]/CS_moments_factor, marker='x',linewidth=0,label="Individual years",color='#1f77b4') 
    i = T
    for t in np.array(range(T-1))+1:
        panel1.plot(empirical_moments_inc[i:min(i+T-t,i+3)], marker='x',linewidth=0,color='#1f77b4')
        panel2.plot(np.array(range(T-t)),empirical_moments_inc[i:i+T-t], marker='x',linewidth=0,color='#1f77b4')
        #panel3.plot(CS_Ndiff[0:min(T-t,3)],CS_moments[i:min(i+T-t,i+3)]/CS_moments_factor[0:min(T-t,3)], marker='x',linewidth=0,color='#1f77b4')
        panel4.plot(CS_Ndiff[0:T-t],CS_moments[i:i+T-t]/CS_moments_factor[0:T-t], marker='x',linewidth=0,color='#1f77b4')
        i += T-t
    panel1.set_title('Variance and\n First Covariance', fontsize=17)
    panel2.set_title('Covariance $(\Delta y_t, \Delta y_{t+n})$ - BPP (2008)', fontsize=17)
    #panel3.set_title('First Few Variances', fontsize=17)
    panel4.set_title('Var$(\Delta^n y)/(n-1/3)$ - Carroll Samwick (1998)', fontsize=17)
    
    panel1.set_xlabel("Time Difference (n)", fontsize=15)
    panel2.set_xlabel("Time Difference (n)", fontsize=15)
    #panel3.set_xlabel("Time Difference (n)", fontsize=15)
    panel4.set_xlabel("Time Difference (n)", fontsize=15)
    
    panel1.set_ylabel("Covariance", fontsize=15)
    #panel2.set_ylabel("Covariance", fontsize=12)
    #panel3.set_ylabel("Variance", fontsize=15)
    panel4.set_ylabel("Var$(\Delta^n y)/(n-1/3)$", fontsize=15)
    
    #plot user defined
    if (half_life!=0.0):
        omega = np.log(2)/half_life
    else:
        omega = 1.0 # use dummy value for omega, and replace with a 'bonus' type shock instead - this is equivalent to zero half life
        bonus = 1.0
    user_params = np.array([perm_var,tran_var,omega,bonus,perm_decay])
    user_cov_full = implied_inc_cov_composite(user_params,T)
    user_cov = user_cov_full[0:T]
    user_CS_moments = CS_from_BPP(user_cov_full)[0:T]   
    
    user_panel1, = panel1.plot(user_cov[0:3], color="orange")
    user_panel2, = panel2.plot(user_cov, color="orange", label='User')
    panel4.plot(CS_Ndiff,user_CS_moments/CS_moments_factor, color="orange", label='User')
    #comparison graph
    if (compare=="All Households"):
        mean_compare_moments = mean_moments
        CS_mean_compare_moments = CS_moments_mean
        panel1.plot(mean_moments[0:3],color='#1f77b4', marker='o')
        panel2.plot(mean_moments,label="Compare To",color='#1f77b4', marker='o')
        panel4.plot(CS_Ndiff,CS_moments_mean/CS_moments_factor,label="Compare To",color='#1f77b4', marker='o')
        panel2.plot(empirical_moments_inc[0:T],color='#1f77b4', linewidth=0, label=' ')
        panel4.plot(CS_Ndiff,CS_moments[0:T]/CS_moments_factor, linewidth=0,color='#1f77b4', label=' ')
        quantile_widget.options=['1']
    else:
        compare_moments_inc, compare_Omega_inc  = compare_to_moments(compare, quantile)
        CS_moments_compare = CS_from_BPP(compare_moments_inc)
        mean_compare_moments = np.zeros(T)
        mean_compare_moments_se = np.zeros(T)
        CS_mean_compare_moments = np.zeros(T)
        for t in range(T):
            this_diag = np.diag(1.0/(T-t)*np.ones(T-t),-t)
            this_diag = this_diag[vech_indicesT]
            mean_compare_moments[t] = np.dot(this_diag,compare_moments_inc)
            mean_compare_moments_se[t] = np.dot(np.dot(this_diag,compare_Omega_inc),this_diag)**0.5
            CS_mean_compare_moments[t] = np.dot(this_diag,CS_moments_compare)
        panel1.plot(mean_compare_moments[0:3],color='#e377c2',marker='o')
        panel2.plot(mean_compare_moments,label="Compare To",color='#e377c2',marker='o')
        panel4.plot(CS_Ndiff,CS_mean_compare_moments/CS_moments_factor,label="Compare To",color='#e377c2',marker='o')
        #plot the moments for each year
        panel1.plot(compare_moments_inc[0:3], marker='x',linewidth=0,color='#e377c2')
        panel2.plot(np.array(range(T)),compare_moments_inc[0:T], marker='x',linewidth=0,label="Individual years",color='#e377c2')
        panel4.plot(CS_Ndiff[0:T],CS_moments_compare[0:T]/CS_moments_factor, marker='x',linewidth=0,label="Individual years",color='#e377c2')
        i = T
        for t in np.array(range(T-1))+1:
            panel1.plot(compare_moments_inc[i:min(i+T-t,i+3)], marker='x',linewidth=0,color='#e377c2')
            panel2.plot(np.array(range(T-t)),compare_moments_inc[i:i+T-t], marker='x',linewidth=0,color='#e377c2')
            panel4.plot(CS_Ndiff[0:T-t],CS_moments_compare[i:i+T-t]/CS_moments_factor[0:T-t], marker='x',linewidth=0,color='#e377c2')
            i += T-t
            # Standard errors
        panel1.plot(mean_compare_moments[0:3]+1.96*mean_compare_moments_se[0:3],linestyle="--",color="gray",linewidth=1.0)
        panel1.plot(mean_compare_moments[0:3]-1.96*mean_compare_moments_se[0:3],linestyle="--",color="gray",linewidth=1.0)
        panel2.plot(mean_compare_moments+1.96*mean_compare_moments_se,linestyle="--",color="gray",linewidth=1.0)
        panel2.plot(mean_compare_moments-1.96*mean_compare_moments_se,linestyle="--",color="gray",linewidth=1.0)
    panel2.legend(loc='lower right', prop={'size': 12}, ncol=2, frameon=False)
    panel4.legend(loc='lower left', prop={'size': 12}, ncol=2, frameon=False)
    panel1.axhline(y=0, color='k',linewidth=1.0)
    panel2.axhline(y=0, color='k',linewidth=1.0)
    #panel3.axhline(y=0, color='k',linewidth=1.0)
    panel4.axhline(y=0, color='k',linewidth=1.0)
    
    panel1.set_ylim(np.array([-0.0025,np.max(np.array([0.0125,1.1*mean_compare_moments[0]]))]))
    panel2.set_ylim(np.array([np.min(np.array([-0.0013,1.1*mean_compare_moments[1]])),0.0003]))
    panel4.set_ylim(np.array([0.0,np.max(np.array([0.02,1.1*CS_mean_compare_moments[0]/CS_moments_factor[0]]))]))


# %% {"code_folding": [0]}
#set up widgets with default values for plot
cont_update = False
orientation = 'vertical'
slider_height = 'auto'
perm_var_widget = widgets.FloatSlider(
    value=estimates[0],
    min=0,
    max=0.02,
    step=0.001,
    description='Perm Var',
    disabled=False,
    continuous_update=cont_update,
    orientation=orientation,
    readout=True,
    readout_format='.4f',
    layout=widgets.Layout(height = slider_height, grid_area='perm_var')
)
tran_var_widget = widgets.FloatSlider(
    value=estimates[1],
    min=0,
    max=0.02,
    step=0.001,
    description='Tran Var',
    disabled=False,
    continuous_update=cont_update,
    orientation=orientation,
    readout=True,
    readout_format='.4f',
    layout=widgets.Layout(height = slider_height, grid_area='tran_var')
)
half_life_widget = widgets.FloatSlider(
    value=np.log(2)/estimates[2],
    min=0,
    max=5.0,
    step=0.1,
    description='Half Life',
    disabled=False,
    continuous_update=cont_update,
    orientation=orientation,
    readout=True,
    readout_format='.1f',
    layout=widgets.Layout(height = slider_height, grid_area='haf_life')
)
bonus_widget = widgets.FloatSlider(
    value=estimates[3],
    min=0,
    max=1.0,
    step=0.05,
    description='Bonus',
    disabled=False,
    continuous_update=cont_update,
    orientation=orientation,
    readout=True,
    readout_format='.2f',
    layout=widgets.Layout(height = slider_height, grid_area='bonus___')
)
perm_decay_widget = widgets.FloatSlider(
    value=estimates[4].astype(bool),
    min=-0.1,
    max=0.1,
    step=0.0001,
    description='Perm Decay',
    disabled=False,
    continuous_update=cont_update,
    orientation=orientation,
    readout=True,
    readout_format='.4f',
    layout=widgets.Layout(height = slider_height, grid_area='perm_dec')
)
estimate_button = widgets.Button(description="Estimate!",
                                 layout=widgets.Layout(width='70%', height='30px', grid_area='est_butt'),
                                 justify_self = 'center')
def estimate_button_clicked(b):
    optimize_index = np.array([not fix_perm.value,not fix_tran.value,not fix_half_life.value,not fix_bonus.value,not fix_perm_decay.value])
    init_params_with_fix = init_params
    slider_values = np.array([perm_var_widget.value, tran_var_widget.value, np.log(2)/half_life_widget.value, bonus_widget.value, perm_decay_widget.value])
    init_params_with_fix[np.logical_not(optimize_index)] = slider_values[np.logical_not(optimize_index)]
    compare = compare_widget.value
    quantile = quantile_widget.value
    compare_moments_inc, compare_Omega_inc  = compare_to_moments(compare, quantile)
    estimates, estimate_se = parameter_estimation(compare_moments_inc, compare_Omega_inc, T, init_params_with_fix, bounds=bounds, optimize_index=optimize_index)  
    perm_var_widget.value = estimates[0]
    tran_var_widget.value = estimates[1]
    half_life_widget.value = np.log(2)/estimates[2]
    bonus_widget.value = estimates[3]
    perm_decay_widget.value = estimates[4]
estimate_button.on_click(estimate_button_clicked)

compare_widget = widgets.Dropdown(
    options=['All Households',
             'Liquid Wealth (quintiles)',
             'Net Wealth (quintiles)',
             'Income (deciles)', 
             'Net Nominal Position (deciles)',
             'Interest Rate Exposure (deciles)',
             'Consumption (deciles)'],
    value='All Households',
    description='Compare To',
    disabled=False,
    layout=widgets.Layout(width='auto', height='auto', grid_area='comp_box')
)
quantiles = ['1']
quantile_widget = widgets.Dropdown(
    options=quantiles,
    value='1',
    description='Quantile',
    disabled=False,
    layout=widgets.Layout(width='auto', height='auto', grid_area='quan_box')
)
graph_update = widgets.interactive(plot_moments,
                                   perm_var=perm_var_widget,
                                   tran_var=tran_var_widget,
                                   half_life=half_life_widget,
                                   bonus=bonus_widget,
                                   perm_decay=perm_decay_widget,
                                   compare=compare_widget,
                                   quantile=quantile_widget
)
# control_widget=widgets.TwoByTwoLayout(
#           bottom_left=estimate_button,
#           top_right=compare_widget,
#           bottom_right=quantile_widget
# )
# slider_widget=widgets.TwoByTwoLayout(
#           top_left=perm_var_widget,
#           top_right=tran_var_widget,
#           bottom_left = half_life_widget,
#           bottom_right=bonus_widget
# )
fix_perm = widgets.Checkbox(
    value=False,#optimize_index[4],
    description='',
    disabled=False,
    indent=False,
    layout=widgets.Layout(width='auto', height='auto')
)
fix_perm_box = widgets.HBox(children=[fix_perm],layout=widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='100%',
                grid_area = 'fix_perm'))
fix_tran = widgets.Checkbox(
    value=False,#optimize_index[4],
    description='',
    disabled=False,
    indent=False,
    layout=widgets.Layout(width='auto', height='auto')
)
fix_tran_box = widgets.HBox(children=[fix_tran],layout=widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='100%',
                grid_area = 'fix_tran'))
fix_half_life = widgets.Checkbox(
    value=False,#optimize_index[4],
    description='',
    disabled=False,
    indent=False,
    layout=widgets.Layout(width='auto', height='auto')
)
fix_half_life_box = widgets.HBox(children=[fix_half_life],layout=widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='100%',
                grid_area = 'fix_half'))
fix_bonus = widgets.Checkbox(
    value=False,#optimize_index[4],
    description='',
    disabled=False,
    indent=False,
    layout=widgets.Layout(width='auto', height='auto')
)
fix_bonus_box = widgets.HBox(children=[fix_bonus],layout=widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='100%',
                grid_area = 'fix_bonu'))
fix_perm_decay = widgets.Checkbox(
    value=True,#optimize_index[4],
    description='',
    disabled=False,
    indent=False,
    layout=widgets.Layout(width='auto', height='auto')
)
fix_perm_decay_box = widgets.HBox(children=[fix_perm_decay],layout=widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='100%',
                grid_area = 'fix_pdec'))

button_box_layout = widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='100%',
                grid_area = 'esti_box')
estimate_box = widgets.HBox(children=[estimate_button],layout=button_box_layout)

empty  = widgets.Button(description='',
                 layout=widgets.Layout(width='auto', grid_area='empty___'),
                 style=widgets.ButtonStyle(button_color='white'))
Fix  = widgets.Button(description='Fix?',
                 layout=widgets.Layout(width='auto', grid_area='fix_____', align_content='flex-end'),
                 style=widgets.ButtonStyle(button_color='white'))

control_panel = widgets.GridBox(children=[empty,
                          compare_widget,
                          quantile_widget,
                          estimate_box,
                          perm_var_widget,
                          tran_var_widget,
                          half_life_widget,
                          bonus_widget,
                          perm_decay_widget,
                          Fix,
                          fix_perm_box,
                          fix_tran_box,
                          fix_half_life_box,
                          fix_bonus_box,
                          fix_perm_decay_box],
        layout=widgets.Layout(
            width='90%',
            grid_template_rows='auto auto auto',
            grid_template_columns='35% 10% 10% 10% 10% 10% 10%',
            grid_template_areas='''
            "comp_box fix_____ fix_perm fix_tran fix_half fix_bonu fix_pdec"
            "quan_box empty___ perm_var tran_var haf_life bonus___ perm_dec"
            "esti_box empty___ perm_var tran_var haf_life bonus___ perm_dec"
            ''')
       )


# %% {"code_folding": [0]}
# Define plotting function for parameters by quantiles
def plot_by_subgroup(subgroup_stub, T, init_params, optimize_index=optimize_index, bounds=bounds):
    subgroup_names = []
    if (subgroup_stub=="Liquid Wealth (quintiles)"):
        subgroup_stub='moments_by_liquid_wealth_quantile'
        num_quantiles=5
    elif (subgroup_stub=="Net Wealth (quintiles)"):
        subgroup_stub='moments_by_net_wealth_quantile'
        num_quantiles=5
    elif (subgroup_stub=="Income (deciles)"):
        subgroup_stub='moments_by_Income_quantile'
        num_quantiles=10
    elif (subgroup_stub=="Net Nominal Position (deciles)"):
        subgroup_stub='moments_by_NNP_quantile'
        num_quantiles=10
    elif (subgroup_stub=="Interest Rate Exposure (deciles)"):
        subgroup_stub='moments_by_URE_quantile'
        num_quantiles=10
    elif (subgroup_stub=="Consumption (deciles)"):
        subgroup_stub='moments_by_MeanCons_quantile'
        num_quantiles=10
    for i in range(num_quantiles):
        subgroup_names += ["X"+str(i+1)]
    estimates, standard_errors = parameter_estimation_by_subgroup(moments_BPP_dir,subgroup_stub,subgroup_names, T, init_params, optimize_index=optimize_index, bounds=bounds)
    fig = plt.figure(figsize=(14, 7),constrained_layout=True)
    fig.figsize=(20,40)
    gs = fig.add_gridspec(2, 2)
    panel1 = fig.add_subplot(gs[0, 0])
    panel2 = fig.add_subplot(gs[0, 1])
    panel3 = fig.add_subplot(gs[1, 0])
    panel4 = fig.add_subplot(gs[1, 1])
    panel1.bar(np.array(range(num_quantiles))+1,estimates[:,0])
    panel2.bar(np.array(range(num_quantiles))+1,estimates[:,1])
    panel3.bar(np.array(range(num_quantiles))+1,np.log(2)/estimates[:,2])
    panel4.bar(np.array(range(num_quantiles))+1,estimates[:,3])
    panel1.set_title("Permanent Variance")
    panel2.set_title("Transitory Variance")
    panel3.set_title("Half-life of Somewhat Transitory Shock")
    panel4.set_title("Share that is Completely Transitory")
    panel1.set_xlabel("Quantile")
    panel2.set_xlabel("Quantile")
    panel3.set_xlabel("Quantile")
    panel4.set_xlabel("Quantile")
    
subgroup_widget = widgets.Dropdown(
    options=['Liquid Wealth (quintiles)',
             'Net Wealth (quintiles)',
             'Income (deciles)', 
             'Net Nominal Position (deciles)',
             'Interest Rate Exposure (deciles)',
             'Consumption (deciles)'],
    value='Income (deciles)',
    description='Subgroup',
    disabled=False,
)


# %% {"code_folding": [0]}
# Plot BPP and Carroll Samwick moments, with estimates and user-defined parameters
# display(control_widget)
# display( slider_widget )
display(control_panel)
graph_update.update()
graph_update.children[7]

# %% {"code_folding": [0]}
# Plot parameter estimates by selected quantiles
widgets.interact(plot_by_subgroup,subgroup_stub=subgroup_widget, T=widgets.fixed(T), init_params=widgets.fixed(init_params), optimize_index=widgets.fixed(optimize_index), bounds=widgets.fixed(bounds));


# %%
