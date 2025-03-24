import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, gamma, norm, multivariate_normal, pearsonr
import copy
import time

psi_11 = np.load('output_2in1/psi_11_steady.npy')
psi_12 = np.load('output_2in1/psi_12_steady.npy')
psi_13 = np.load('output_2in1/psi_13_steady.npy')
psi_21 = np.load('output_2in1/psi_21_steady.npy')
psi_22 = np.load('output_2in1/psi_22_steady.npy')
psi_23 = np.load('output_2in1/psi_23_steady.npy')

B_psi = np.load('input/B_psi_91.npy')
shape_psi = 5

psi_11_hat = psi_11[-1]
psi_12_hat = psi_12[-1]
psi_13_hat = psi_13[-1]
psi_21_hat = psi_21[-1]
psi_22_hat = psi_22[-1]
psi_23_hat = psi_23[-1]

t11_ev = B_psi@psi_11_hat
t12_ev = B_psi@psi_12_hat
t13_ev = B_psi@psi_13_hat
t21_ev = B_psi@psi_21_hat
t22_ev = B_psi@psi_22_hat
t23_ev = B_psi@psi_23_hat

num1 = 1.01
num2 = 0.99
num3 = 2
num4 = 0
# plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)

psi_11_lengend = [r'$\psi_{11_%d}$'%(i+1) for i in range(shape_psi)]
psi_12_lengend = [r'$\psi_{12_%d}$'%(i+1) for i in range(shape_psi)]
psi_13_lengend = [r'$\psi_{13_%d}$'%(i+1) for i in range(shape_psi)]

# temp with node
import datetime
now = datetime.datetime.now()

Max_t = 720
font = {'family': 'Times New Roman',
        'style': 'normal',
        'weight': 'normal',
        'color': 'black',
        'size': 25}
xtick = np.linspace(0, Max_t, 12+1, endpoint=True)
xplot = np.linspace(0, Max_t, B_psi.shape[0], endpoint=True)
xnode = np.linspace(0, Max_t, B_psi.shape[1], endpoint=True)
fig11, ax11 = plt.subplots(1,1,figsize=(9,6))

ax11.plot(xplot, t11_ev, label=r'$\theta_{11}$', linewidth=2, color='#5cb200', linestyle='-')
ax11.plot(xplot, t12_ev, label=r'$\theta_{12}$', linewidth=2, color='#fb7d07', linestyle='--')
ax11.plot(xplot, t13_ev, label=r'$\theta_{13}$', linewidth=2, color='#2976bb', linestyle='-.')

ax11.set_xlim(0, 720)
# ax11.set_ylim(7, 12)
ax11.set_xlabel('Time/Hour', fontdict=font)
ax11.set_ylabel('Time-varying parameters', fontdict=font)

ax11.set_xticks(xtick)
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
                 labelsize=20)

ax11.grid(linestyle='--')
bwith = 0.5 

ax11.spines['bottom'].set_linewidth(bwith)
ax11.spines['left'].set_linewidth(bwith)
ax11.spines['top'].set_linewidth(bwith)
ax11.spines['right'].set_linewidth(bwith)

ax11.spines['bottom'].set_color('black')
ax11.spines['left'].set_color('black')
ax11.spines['top'].set_color('black')
ax11.spines['right'].set_color('black')

plt.legend(ncol=1, fontsize=18, bbox_to_anchor=(0.025, num2), loc=num3, borderaxespad=num4)
plt.tight_layout()
plt.savefig('pic/Time-varying parameters-Temperature-%s-2in1'%(now.strftime('%Y%m%d')), dpi=1200, bbox_inches='tight')

# hum with node
fig12, ax12 = plt.subplots(1,1,figsize=(9,6))

ax12.plot(xplot, t21_ev, label=r'$\theta_{21}$', linewidth=2, color='#5cb200', linestyle='-')
ax12.plot(xplot, t22_ev, label=r'$\theta_{22}$', linewidth=2, color='#fb7d07', linestyle='--')
ax12.plot(xplot, t23_ev, label=r'$\theta_{23}$', linewidth=2, color='#2976bb', linestyle='-.')

ax12.set_xlim(0, 720)
ax12.set_ylim(0, 0.0008)

ax12.set_xlabel('Time/Hour', fontdict=font)
ax12.set_ylabel('Time-varying parameters', fontdict=font)

ax12.set_xticks(xtick)
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
                 labelsize=20)

ax12.grid(linestyle='--')
bwith = 0.5 

ax12.spines['bottom'].set_linewidth(bwith)
ax12.spines['left'].set_linewidth(bwith)
ax12.spines['top'].set_linewidth(bwith)
ax12.spines['right'].set_linewidth(bwith)

ax12.spines['bottom'].set_color('black')
ax12.spines['left'].set_color('black')
ax12.spines['top'].set_color('black')
ax12.spines['right'].set_color('black')

plt.legend(ncol=1, fontsize=18, bbox_to_anchor=(0.025, num2), loc=num3, borderaxespad=num4)
plt.tight_layout()
plt.savefig('pic/Time-varying parameters-Humidity-%s-2in1'%(now.strftime('%Y%m%d')), dpi=1200, bbox_inches='tight')