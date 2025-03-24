import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import  ConnectionPatch
from scipy.stats import invgamma, gamma, norm, multivariate_normal
import copy
from time import perf_counter

B_psi = np.load('input_penalty_cut/B_psi.npy')
shape_psi = 5

# settings of Trace plot
num1 = 1.01
num2 = 0.99
num3 = 2
num4 = 0

N_1 = 5000
# psi 1 legend
psi_11_lengend = [r'$\psi_{11_%d}$'%(i+1) for i in range(shape_psi)]
psi_12_lengend = [r'$\psi_{12_%d}$'%(i+1) for i in range(shape_psi)]
psi_13_lengend = [r'$\psi_{13_%d}$'%(i+1) for i in range(shape_psi)]
# psi 2 
psi_21_lengend = [r'$\psi_{21_%d}$'%(i+1) for i in range(shape_psi)]
psi_22_lengend = [r'$\psi_{22_%d}$'%(i+1) for i in range(shape_psi)]
psi_23_lengend = [r'$\psi_{23_%d}$'%(i+1) for i in range(shape_psi)]
# psi 3 
psi_31_lengend = [r'$\psi_{31_%d}$'%(i+1) for i in range(shape_psi)]
psi_32_lengend = [r'$\psi_{32_%d}$'%(i+1) for i in range(shape_psi)]
psi_33_lengend = [r'$\psi_{33_%d}$'%(i+1) for i in range(shape_psi)]

psi_11_hat = np.load('output_3in1/psi_11_burn.npy')[-1]
psi_12_hat = np.load('output_3in1/psi_12_burn.npy')[-1]
psi_13_hat = np.load('output_3in1/psi_13_burn.npy')[-1]
psi_21_hat = np.load('output_3in1/psi_21_burn.npy')[-1]
psi_22_hat = np.load('output_3in1/psi_22_burn.npy')[-1]
psi_23_hat = np.load('output_3in1/psi_23_burn.npy')[-1]
psi_31_hat = np.load('output_3in1/psi_31_burn.npy')[-1]
psi_32_hat = np.load('output_3in1/psi_32_burn.npy')[-1]
psi_33_hat = np.load('output_3in1/psi_33_burn.npy')[-1]

# plot settings time-varying
Max_t = 40
font = {'family': 'Times new Roman',
        'style': 'normal',
        'weight': 'normal',
        'color': 'black',
        'size': 25}
xtick = np.linspace(0, Max_t, 8+1, endpoint=True) # time 9 node
xplot = np.linspace(0, Max_t, B_psi.shape[0], endpoint=True) # time 41
xnode = np.linspace(0, Max_t, B_psi.shape[1], endpoint=True) # parameter
psi_node = np.load('input_penalty_cut/psi_node.npy')
lennode = shape_psi
xpsi = np.linspace(0, B_psi.shape[0] - 1, lennode) #0-40

t11_ev = B_psi@psi_11_hat
t12_ev = B_psi@psi_12_hat
t13_ev = B_psi@psi_13_hat

t21_ev = B_psi@psi_21_hat
t22_ev = B_psi@psi_22_hat
t23_ev = B_psi@psi_23_hat

t31_ev = B_psi@psi_31_hat
t32_ev = B_psi@psi_32_hat
t33_ev = B_psi@psi_33_hat

# t11_real = B_psi@psi_node[0][0]
# t12_real = B_psi@psi_node[0][1]
# t13_real = B_psi@psi_node[0][2]
# t21_real = B_psi@psi_node[1][0]
# t22_real = B_psi@psi_node[1][1]
# t23_real = B_psi@psi_node[1][2]
# t31_real = B_psi@psi_node[2][0]
# t32_real = B_psi@psi_node[2][1]
# t33_real = B_psi@psi_node[2][2]
t11_real = psi_node[0][0]
t12_real = psi_node[0][1]
t13_real = psi_node[0][2]
t21_real = psi_node[1][0]
t22_real = psi_node[1][1]
t23_real = psi_node[1][2]
t31_real = psi_node[2][0]
t32_real = psi_node[2][1]
t33_real = psi_node[2][2]

# psi 1 PDE1 without node
import datetime
now = datetime.datetime.now()
fig25, ax25 = plt.subplots(1,1,figsize=(9,6))
xt = np.linspace(0, 40, 5)
ax25.plot(xt, t11_real, 'x', label=r'$\psi_{11}$', markersize=6, color='#ef4026', zorder=3, clip_on=False)
ax25.plot(xt, t12_real, '.', label=r'$\psi_{12}$', markersize=6, color='#009337', zorder=3, clip_on=False)
ax25.plot(xt, t13_real, '*', label=r'$\psi_{13}$', markersize=6, color='#2976bb', zorder=3, clip_on=False)
ax25.plot(t11_ev, label=r'$\theta_{11}$', linewidth=1, color='#fb7d07', linestyle='-')
ax25.plot(t12_ev, label=r'$\theta_{12}$', linewidth=1, color='#5cb200', linestyle='--')
ax25.plot(t13_ev, label=r'$\theta_{13}$', linewidth=1, color='#75bbfd', linestyle='-.')

ax25.set_xlim(0, 40)
ax25.set_ylim(0.02, 0.16)
ax25.tick_params(labelsize=20)
ax25.set_xlabel('Time', fontdict=font)
ax25.set_ylabel('Time-varying parameters', fontdict=font)

ax25.set_xticks(xtick)
ax25.tick_params(axis='both',
                which='both',
                colors='black',
                top='on',
                bottom='on',
                left='on',
                right='on',                
                direction='in',              
                length=5,
                width=0.5,
                 labelsize=20)

ax25.grid(linestyle='--')
bwith = 0.5 
ax25.spines['bottom'].set_linewidth(bwith)
ax25.spines['left'].set_linewidth(bwith)
ax25.spines['top'].set_linewidth(bwith)
ax25.spines['right'].set_linewidth(bwith)

ax25.spines['bottom'].set_color('black')
ax25.spines['left'].set_color('black')
ax25.spines['top'].set_color('black')
ax25.spines['right'].set_color('black')

plt.legend(ncol=2, fontsize=18, bbox_to_anchor=(0.025, num2), loc=num3, borderaxespad=num4)
plt.tight_layout()
plt.savefig('pic/3in1_PDE 1_True&Estimation-%s'%now.strftime('%y%m%d'), dpi=1200, bbox_inches='tight')

# psi 2 PDE2 without node
fig26, ax26 = plt.subplots(1,1,figsize=(9,6))

ax26.plot(xt,t21_real, 'x', label=r'$\psi_{21}$', markersize=6, color='#ef4026', zorder=3, clip_on=False)

ax26.plot(xt, t22_real, '.', label=r'$\psi_{22}$', markersize=6, color='#009337', zorder=3, clip_on=False)

ax26.plot(xt, t23_real, '*', label=r'$\psi_{23}$', markersize=6, color='#2976bb', zorder=3, clip_on=False)

ax26.plot(t21_ev, label=r'$\theta_{21}$', linewidth=1, color='#fb7d07', linestyle='-')

ax26.plot(t22_ev, label=r'$\theta_{22}$', linewidth=1, color='#5cb200', linestyle='--')

ax26.plot(t23_ev, label=r'$\theta_{23}$', linewidth=1, color='#75bbfd', linestyle='-.')

ax26.set_xlim(0, 40)
ax26.set_ylim(0.02, 0.16)
ax26.set_xlabel('Time', fontdict=font)
ax26.set_ylabel('Time-varying parameters', fontdict=font)

ax26.set_xticks(xtick)
ax26.tick_params(axis='both',
                which='both',
                colors='black',
                top='on',
                bottom='on',
                left='on',
                right='on',                
                direction='in',              
                length=5,
                width=0.5,
                 labelsize=20)

ax26.grid(linestyle='--')
bwith = 0.5

ax26.spines['bottom'].set_linewidth(bwith)
ax26.spines['left'].set_linewidth(bwith)
ax26.spines['top'].set_linewidth(bwith)
ax26.spines['right'].set_linewidth(bwith)

ax26.spines['bottom'].set_color('black')
ax26.spines['left'].set_color('black')
ax26.spines['top'].set_color('black')
ax26.spines['right'].set_color('black')

plt.legend(ncol=2, fontsize=18, bbox_to_anchor=(0.025, num2), loc=num3, borderaxespad=num4)
plt.tight_layout()
plt.savefig('pic/3in1_PDE 2_True&Estimation-%s'%now.strftime('%y%m%d'), dpi=1200, bbox_inches='tight')

plt.show()

# psi 3  PDE3 without node
fig27, ax27 = plt.subplots(1, 1, figsize=(9, 6))

ax27.plot(xt, t31_real, 'x', label=r'$\psi_{31}$', markersize=6, color='#ef4026', zorder=3, clip_on=False)

ax27.plot(xt, t32_real, '.', label=r'$\psi_{32}$', markersize=6, color='#009337', zorder=3, clip_on=False)

ax27.plot(xt, t33_real, '*', label=r'$\psi_{33}$', markersize=6, color='#2976bb', zorder=3, clip_on=False)

ax27.plot(t31_ev, label=r'$\theta_{31}$', linewidth=1, color='#fb7d07', linestyle='-')

ax27.plot(t32_ev, label=r'$\theta_{32}$', linewidth=1, color='#5cb200', linestyle='--')

ax27.plot(t33_ev, label=r'$\theta_{33}$', linewidth=1, color='#75bbfd', linestyle='-.')

ax27.set_xlim(0, 40)
ax27.set_ylim(0.02, 0.16)
ax27.set_xlabel('Time', fontdict=font)
ax27.set_ylabel('Time-varying parameters', fontdict=font)

ax27.set_xticks(xtick)
ax27.tick_params(axis='both',
                 which='both',
                 colors='black', 
                 top='on',  
                 bottom='on', 
                 left='on',
                 right='on',
                 direction='in',
                 length=5, 
                 width=0.5,
                 labelsize=20)
ax27.set_xticks(xtick)
ax27.grid(linestyle='--')
bwith = 0.5  
ax27.spines['bottom'].set_linewidth(bwith) 
ax27.spines['left'].set_linewidth(bwith) 
ax27.spines['top'].set_linewidth(bwith) 
ax27.spines['right'].set_linewidth(bwith) 

ax27.spines['bottom'].set_color('black') 
ax27.spines['left'].set_color('black') 
ax27.spines['top'].set_color('black') 
ax27.spines['right'].set_color('black') 

plt.legend(ncol=2, fontsize=18, bbox_to_anchor=(0.025, num2), loc=num3, borderaxespad=num4)
plt.tight_layout()
plt.savefig('pic/3in1_PDE 3_True&Estimation-%s'%now.strftime('%y%m%d'), dpi=1200, bbox_inches='tight')

plt.show()

B1 = np.load('input_penalty_cut/B_mat_TV_1.npy')
beta1_ini = np.load('input_penalty_cut/beta_TV_1.npy')

n = B1.shape[0] #obeservation
K = beta1_ini.shape[0] #basis function
N_1 = 5000

res_beta1 = np.load('output_3in1/beta1_burn.npy')
res_beta2 = np.load('output_3in1/beta2_burn.npy')
res_beta3 = np.load('output_3in1/beta3_burn.npy')

def zone_and_linked(ax,axins,zone_left,zone_right,x,linked='bottom',
                    x_ratio=0.05,y_ratio=0.05):
    """
    axins:      area:axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    x:          X tick
    y:           list of y
    linked:     {'bottom','top','left','right'}
    x_ratio:    X ratio
    y_ratio:    Y ratio
    """
    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    ylim_bottom = 0.37
    ylim_top = 0.6

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)
        
    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)


# In[35]:


font_beta = {'family': 'Times new Roman',
        'style': 'normal',
        'weight': 'normal',
        'color': 'black',
        'size': 15}

# beta 1 trace plot
x = np.arange(0,5000)

beta1_trace1 = np.empty([K,len(res_beta1[:N_1])])
for i in range(len(res_beta1[:N_1])):
    beta1_trace1[:,i] = res_beta1[:N_1][i]
    
fig10, ax10 = plt.subplots(1,1,figsize=(4,3))
for i in range(K):
    ax10.plot(beta1_trace1[i], linewidth=0.5)
ax10.set_xlabel('Iteration', fontdict=font_beta)
ax10.set_ylabel(r'$\beta_{1}$', fontdict=font_beta)
ax10.tick_params(axis='both',
                 which='both',
                 colors='black',  
                 top='on', 
                 bottom='on', 
                 left='on',
                 right='on',
                 direction='in',
                 length=5, 
                 width=0.5,
                 labelsize=12)

ax10ins = ax10.inset_axes((0.475, 0.5, 0.475, 0.475))
ax10ins.set_facecolor ('white')
ax10ins.yaxis.set_major_locator(plt.NullLocator())
ax10ins.xaxis.set_major_locator(plt.NullLocator())

for i in range(int(K/2)):
    if beta1_trace1[i][0] < 0.6:
        ax10ins.plot((beta1_trace1)[i], linewidth=0.5)

# link
zone_and_linked(ax10, ax10ins, 0, 1500, x , 'right')

plt.suptitle(r'Trace plot of $\beta_{1}$ (Burn-in stage)',x=0.6, y=0.9, fontsize=15)
plt.tight_layout()
plt.savefig('pic/3in1_Beta1 trace_burn', dpi=1200, bbox_inches='tight')
plt.show()

beta1_trace2 = np.empty([K,len(res_beta1[N_1:])])
for i in range(len(res_beta1[N_1:])):
    beta1_trace2[:,i] = res_beta1[N_1:][i]

fig11, ax11 = plt.subplots(1,1,figsize=(4,3))
for i in range(K):
    ax11.plot(beta1_trace2[i], linewidth=0.5)
ax11.set_xlabel('Iteration', fontdict=font_beta)
ax11.set_ylabel(r'$\beta_{1}$', fontdict=font_beta)

ax11.tick_params(axis='both',
                 which='both',
                 colors='black', 
                 top='on', 
                 bottom='on',  
                 left='on',
                 right='on',
                 direction='in',
                 length=5, 
                 width=0.5,
                 labelsize=12)
plt.suptitle(r'Trace plot of $\beta_{1}$ (Steady stage)',x=0.6, y=0.9, fontsize=15)
plt.tight_layout()
plt.savefig('pic/3in1_Beta1 trace_steady', dpi=1200, bbox_inches='tight')
plt.show()

# beta 2 trace plot
beta2_trace1 = np.empty([K,len(res_beta2[:N_1])])
for i in range(len(res_beta2[:N_1])):
    beta2_trace1[:,i] = res_beta2[:N_1][i]

fig12, ax12 = plt.subplots(1,1,figsize=(4,3))
for i in range(K):
    ax12.plot(beta2_trace1[i], linewidth=0.5)
ax12.set_xlabel('Iteration', fontdict=font_beta)
ax12.set_ylabel(r'$\beta_{2}$', fontdict=font_beta)
ax12.tick_params(axis='both',
                 which='both',
                 colors='black', 
                 top='on', 
                 bottom='on',  
                 left='on',
                 right='on',
                 direction='in',
                 length=5, 
                 width=0.5,
                 labelsize=12)
ax12ins = ax12.inset_axes((0.475, 0.5, 0.475, 0.475))
ax12ins.set_facecolor ('white')
ax12ins.yaxis.set_major_locator(plt.NullLocator())
ax12ins.xaxis.set_major_locator(plt.NullLocator())

for i in range(int(K/2)):
    if beta2_trace1[i][0] < 0.6:
        ax12ins.plot((beta2_trace1)[i], linewidth=0.5)

# link
zone_and_linked(ax12, ax12ins, 0, 1500, x , 'right')

plt.suptitle(r'Trace plot of $\beta_{2}$ (Burn-in stage)',x=0.6, y=0.9, fontsize=15)
plt.tight_layout()
plt.savefig('pic/3in1_Beta2 trace_burn', dpi=1200, bbox_inches='tight')
plt.show()

beta2_trace2 = np.empty([K,len(res_beta2[N_1:])])
for i in range(len(res_beta2[N_1:])):
    beta2_trace2[:,i] = res_beta2[N_1:][i]

fig13, ax13 = plt.subplots(1,1,figsize=(4,3))
for i in range(K):
    ax13.plot(beta2_trace2[i], linewidth=0.5)
ax13.set_xlabel('Iteration', fontdict=font_beta)
ax13.set_ylabel(r'$\beta_{2}$', fontdict=font_beta)
ax13.tick_params(axis='both',
                 which='both',
                 colors='black',
                 top='on',
                 bottom='on', 
                 left='on',
                 right='on',
                 direction='in',
                 length=5, 
                 width=0.5,
                 labelsize=12)
plt.suptitle(r'Trace plot of $\beta_{2}$ (Steady stage)',x=0.6, y=0.9, fontsize=15)
plt.tight_layout()
plt.savefig('pic/3in1_Beta2 trace_steady', dpi=1200, bbox_inches='tight')
plt.show()

beta3_trace1 = np.empty([K,len(res_beta3[:N_1])])
for i in range(len(res_beta3[:N_1])):
    beta3_trace1[:,i] = res_beta3[:N_1][i]

fig14, ax14 = plt.subplots(1,1,figsize=(4,3))
for i in range(K):
    ax14.plot(beta3_trace1[i], linewidth=0.5)
ax14.set_xlabel('Iteration', fontdict=font_beta)
ax14.set_ylabel(r'$\beta_{3}$', fontdict=font_beta)
ax14.tick_params(axis='both',
                 which='both',
                 colors='black',
                 top='on', 
                 bottom='on',
                 left='on',
                 right='on',
                 direction='in',
                 length=5,
                 width=0.5,
                 labelsize=12)
ax14ins = ax14.inset_axes((0.475, 0.5, 0.475, 0.475))
ax14ins.set_facecolor ('white')
ax14ins.yaxis.set_major_locator(plt.NullLocator())
ax14ins.xaxis.set_major_locator(plt.NullLocator())

for i in range(int(K/2)):
    if beta3_trace1[i][0] < 0.6:
        ax14ins.plot((beta3_trace1)[i], linewidth=0.5)

# link
zone_and_linked(ax14, ax14ins, 0, 1500, x , 'right')

plt.suptitle(r'Trace plot of $\beta_{3}$ (Burn-in stage)',x=0.6, y=0.9, fontsize=15)
plt.tight_layout()
plt.savefig('pic/3in1_Beta3 trace_burn', dpi=1200, bbox_inches='tight')
plt.show()


beta3_trace2 = np.empty([K,len(res_beta3[N_1:])])
for i in range(len(res_beta3[N_1:])):
    beta3_trace2[:,i] = res_beta3[N_1:][i]

fig15, ax15 = plt.subplots(1,1,figsize=(4,3))
for i in range(K):
    ax15.plot(beta3_trace2[i], linewidth=0.5)
ax15.set_xlabel('Iteration', fontdict=font_beta)
ax15.set_ylabel(r'$\beta_{3}$', fontdict=font_beta)
ax15.tick_params(axis='both',
                 which='both',
                 colors='black', 
                 top='on',  
                 bottom='on',
                 left='on',
                 right='on',
                 direction='in',
                 length=5, 
                 width=0.5,
                 labelsize=12)
plt.suptitle(r'Trace plot of $\beta_{3}$ (Steady stage)',x=0.6, y=0.9, fontsize=15)
plt.tight_layout()
plt.savefig('pic/3in1_Beta3 trace_steady', dpi=1200, bbox_inches='tight')
plt.show()