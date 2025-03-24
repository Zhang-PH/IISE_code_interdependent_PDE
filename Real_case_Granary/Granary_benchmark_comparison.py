import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import invgamma, gamma, norm, multivariate_normal, pearsonr

# import seaborn as sns
import copy
import time

df1 = np.load("input/xg-temp1030-l1.npy")
df2 = np.load("input/xg-temp1030-l2.npy")
df3 = np.load("input/xg-temp1030-l3.npy")

df4 = np.load("input/xg-hum1030-l1.npy")
df5 = np.load("input/xg-hum1030-l2.npy")
df6 = np.load("input/xg-hum1030-l3.npy")
t_total = 1769
boundary1 = np.zeros([3, 6, 3, t_total])
boundary2 = np.zeros([3, 6, 3, t_total])
for t in range(t_total):
    for i in range(3):
        for j in range(6):
            boundary1[i][j][0][t] = df1[t][6 * i + j + 1]
            boundary1[i][j][1][t] = df2[t][6 * i + j + 1]
            boundary1[i][j][2][t] = df3[t][6 * i + j + 1]
            boundary2[i][j][0][t] = df4[t][6 * i + j + 1]
            boundary2[i][j][1][t] = df5[t][6 * i + j + 1]
            boundary2[i][j][2][t] = df6[t][6 * i + j + 1]


Y_t = np.load('input/Y_temp.npy')
B_t = np.load('input/B_mat_temp.npy')

beta_ini_t = np.load('input/beta_temp.npy')
Y_hat_t = B_t@beta_ini_t

Y_h = np.load('input/Y_hum.npy')
B_h = np.load('input/B_mat_hum.npy')# B_h = B_t

beta_ini_h = np.load('input/beta_hum.npy')
Y_hat_h = B_h@beta_ini_h

# interactive
beta_temp_1 = np.load('output_2in1/beta1_steady.npy')[-1]
beta_hum_1 = np.load('output_2in1/beta2_steady.npy')[-1]

# independent
beta_temp_2 = np.load('output_temp/beta_steady.npy')[-1]
beta_hum_2 = np.load('output_hum/beta_steady.npy')[-1]

# constant
beta_temp_3 = np.load('output_temp_constant/beta_steady.npy')[-1]
beta_hum_3 = np.load('output_hum_constant/beta_steady.npy')[-1]

# tvp-pcm
beta_temp_4 = np.load('output_cascading/beta_temp.npy')
beta_hum_4 = np.load('output_cascading/beta_hum.npy')

# cp-pcm
beta_temp_5 = np.load('output_cascading_constant/beta_temp.npy')
beta_hum_5 = np.load('output_cascading_constant/beta_hum.npy')


temp_1 = B_t@beta_temp_1
temp_2 = B_t@beta_temp_2
temp_3 = B_t@beta_temp_3
temp_4 = B_t@beta_temp_4
temp_5 = B_t@beta_temp_5

hum_1 = B_h@beta_hum_1
hum_2 = B_h@beta_hum_2
hum_3 = B_h@beta_hum_3
hum_4 = B_h@beta_hum_4
hum_5 = B_h@beta_hum_5

tp = 91
xp = 3
yp = 6
zp = 3

temp_restore_1 = np.zeros([xp, yp, zp, tp])# unknown time point
temp_restore_2 = np.zeros([xp, yp, zp, tp])# unknown time point
temp_restore_3 = np.zeros([xp, yp, zp, tp])# unknown time point
temp_restore_4 = np.zeros([xp, yp, zp, tp])# unknown time point
temp_restore_5 = np.zeros([xp, yp, zp, tp])# unknown time point
hum_restore_1 = np.zeros([xp, yp, zp, tp])# unknown time point
hum_restore_2 = np.zeros([xp, yp, zp, tp])# unknown time point
hum_restore_3 = np.zeros([xp, yp, zp, tp])# unknown time point
hum_restore_4 = np.zeros([xp, yp, zp, tp])# unknown time point
hum_restore_5 = np.zeros([xp, yp, zp, tp])# unknown time point

for i in range(xp):
    for j in range(yp):
        for k in range(zp):
            for l in range(tp):
                temp_restore_1[i, j, k, l]= temp_1[tp * (zp * (yp * i + j) + k) + l]
                temp_restore_2[i, j, k, l]= temp_2[tp * (zp * (yp * i + j) + k) + l]
                temp_restore_3[i, j, k, l]= temp_3[tp * (zp * (yp * i + j) + k) + l]
                temp_restore_4[i, j, k, l]= temp_4[tp * (zp * (yp * i + j) + k) + l]
                temp_restore_5[i, j, k, l]= temp_5[tp * (zp * (yp * i + j) + k) + l]
                hum_restore_4[i, j, k, l]= hum_4[tp * (zp * (yp * i + j) + k) + l]
                hum_restore_5[i, j, k, l]= hum_5[tp * (zp * (yp * i + j) + k) + l]
                hum_restore_1[i, j, k, l]= hum_1[tp * (zp * (yp * i + j) + k) + l]
                hum_restore_2[i, j, k, l]= hum_2[tp * (zp * (yp * i + j) + k) + l]
                hum_restore_3[i, j, k, l]= hum_3[tp * (zp * (yp * i + j) + k) + l]

plt.rcParams['font.family'] = 'Times New Roman'

import datetime
now = datetime.datetime.now()

Max_t = 720
Max_x = 9.8
Max_y = 24.5
Max_z = 3.8
rhot_new = 8
tt = np.linspace(0, Max_t, tp)
font = {'family': 'Times New Roman',
        'style': 'normal',
        'weight': 'normal',
        'color': 'black',
        'size': 25}
xtick = np.linspace(0, Max_t, 12+1, endpoint=True)
fig1, ax1 = plt.subplots(1,1,figsize=(9,6))

index = 1#-------------------------index 123

if index == 1:
    x1_point = 0
    y1_point = 0
    z1_point = 2
elif index == 2:
    x1_point = 1
    y1_point = 2
    z1_point = 0
elif index == 3:
    x1_point = 2
    y1_point = 4
    z1_point = 1

ax1.plot(tt, boundary1[x1_point, y1_point, z1_point, 0:(Max_t+1):rhot_new], '+', color='#0343df', label='(%.1f, %.1f, %.1f) True value'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), markersize=6, zorder=2, clip_on=False)

ax1.plot(tt, temp_restore_1[x1_point, y1_point, z1_point, :] , label='(%.1f, %.1f, %.1f) IntPDE-TVP'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), linewidth=1, color='#f7022a', linestyle='-', zorder=5)
ax1.plot(tt, temp_restore_2[x1_point, y1_point, z1_point, :] , label='(%.1f, %.1f, %.1f) IndPDE-TVP'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), linewidth=1, color='#fb7d07', linestyle='-.', zorder=4)
ax1.plot(tt, temp_restore_3[x1_point, y1_point, z1_point, :] , label='(%.1f, %.1f, %.1f) IndPDE-CP'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), linewidth=1, color='#02c14d', linestyle='--', zorder=3)
ax1.plot(tt, temp_restore_4[x1_point, y1_point, z1_point, :] , label='(%.1f, %.1f, %.1f) IndPDE-TVP'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), linewidth=1, color='#8e6ef0', linestyle='-.', zorder=2)
ax1.plot(tt, temp_restore_5[x1_point, y1_point, z1_point, :] , label='(%.1f, %.1f, %.1f) IndPDE-CP'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), linewidth=1, color='#ffdf40', linestyle='--', zorder=1)


plt.tight_layout()
ax1.set_xlim(0, 720)
# ax1.set_ylim(7, 11.5)
ax1.set_xlabel('Time/Hour', fontdict=font)
ax1.set_ylabel('Temperature/℃', fontdict=font)

ax1.set_xticks(xtick)
ax1.tick_params(axis='both',
                which='both',
                colors='black',
                top='on',
                bottom='on',
                left='on',
                right='on',                
                direction='in',              
                length=5,
                width=0.5,
                 labelsize=18)

ax1.grid(linestyle='--')
bwith = 0.5 

ax1.spines['bottom'].set_linewidth(bwith)
ax1.spines['left'].set_linewidth(bwith)
ax1.spines['top'].set_linewidth(bwith)
ax1.spines['right'].set_linewidth(bwith)

ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.spines['top'].set_color('black')
ax1.spines['right'].set_color('black')
plt.tight_layout()
plt.legend(ncol=1, loc='upper left', fontsize=18)

plt.savefig('pic/Temperature_Comparison-%s-%d'%(now.strftime('%Y%m%d'), index), dpi=1200)

fig2, ax2 = plt.subplots(1,1,figsize=(9,6))

# x1_point = 0
# y1_point = 0
# z1_point = 2
# x2_point = 1
# y2_point = 3
# z2_point = 1
# x3_point = 2
# y3_point = 4
# z3_point = 0
ax2.plot(tt, boundary2[x1_point, y1_point, z1_point, 0:(Max_t+1):rhot_new], '+', color='#0343df', label='(%.1f, %.1f, %.1f) True value'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), markersize=6, zorder=2, clip_on=False)

ax2.plot(tt, hum_restore_1[x1_point, y1_point, z1_point, :] , label='(%.1f, %.1f, %.1f) IntPDE-TVP'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), linewidth=1, color='#f7022a', linestyle='-', zorder=5)
ax2.plot(tt, hum_restore_2[x1_point, y1_point, z1_point, :] , label='(%.1f, %.1f, %.1f) IndPDE-TVP'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), linewidth=1, color='#fb7d07', linestyle='-.', zorder=4)
ax2.plot(tt, hum_restore_3[x1_point, y1_point, z1_point, :] , label='(%.1f, %.1f, %.1f) IndPDE-CP'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), linewidth=1, color='#02c14d', linestyle='--', zorder=3)
ax2.plot(tt, hum_restore_4[x1_point, y1_point, z1_point, :] , label='(%.1f, %.1f, %.1f) IndPDE-TVP'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), linewidth=1, color='#8e6ef0', linestyle='-.', zorder=2)
ax2.plot(tt, hum_restore_5[x1_point, y1_point, z1_point, :] , label='(%.1f, %.1f, %.1f) IndPDE-CP'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), linewidth=1, color='#ffdf40', linestyle='--', zorder=1)


ax2.set_xlim(0, 720)

ax2.set_xlabel('Time/Hour', fontdict=font)
ax2.set_ylabel('Humidity/%', fontdict=font)

ax2.set_xticks(xtick)
ax2.tick_params(axis='both',
                which='both',
                colors='black',
                top='on',
                bottom='on',
                left='on',
                right='on',                
                direction='in',              
                length=5,
                width=0.5,
                 labelsize=18)

ax2.grid(linestyle='--')
bwith = 0.5

ax2.spines['bottom'].set_linewidth(bwith)
ax2.spines['left'].set_linewidth(bwith)
ax2.spines['top'].set_linewidth(bwith)
ax2.spines['right'].set_linewidth(bwith)

ax2.spines['bottom'].set_color('black')
ax2.spines['left'].set_color('black')
ax2.spines['top'].set_color('black')
ax2.spines['right'].set_color('black')

plt.legend(ncol=1, loc='upper left', fontsize=18)
plt.tight_layout()
plt.savefig('pic/Humidity_Comparison-%s-%d'%(now.strftime('%Y%m%d'), index), dpi=1200)