import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, gamma, norm, multivariate_normal, pearsonr
import copy
import random
import time
import openpyxl

#Bspline
class Bspline:
    def __init__(self):
        self.fitted = {}
        self.settings = {}

    def Base(self, u, i, p, knot):
        """
        ith B-Spline basis function of order p.
        param u: 1d input
        param i: spline function index
        param p: order of spline(order = degree + 1)
        param knot: knot sequence
        """
        self.tau = knot# tau:inner knots+ side knot
        if p == 0:
            if (knot[i] <= u) and (u < knot[i + 1]):
                res = 1
            else:
                res = 0
        else:
            c1, c2 = 0, 0
            len1 = knot[i + p] - knot[i]
            len2 = knot[i + p + 1] - knot[i + 1]
            if len1==0:
                c1 = 0
            else:
                c1 = (u - knot[i]) / (knot[i + p] - knot[i]) * self.Base(u, i, p - 1, knot)
            if len2==0:
                c2 = 0
            else:
                c2 = (knot[i + p + 1] - u) / (knot[i + p + 1] - knot[i + 1]) * self.Base(u, i + 1, p - 1, knot)
            res = c1 + c2
        return res
    
    def Base2(self, u, i, p, knot):
        """
        ith B-Spline basis function of order p.
        param u: 1d input
        param i: spline function index
        param p: order of spline(order = degree + 1)
        param knot: knot sequence
        """
        self.tau = knot
        if p == 0:
            if (knot[i] <= u) and (u <= knot[i + 1]):
                res = 1
            else:
                res = 0
        else:
            c1, c2 = 0, 0
            len1 = knot[i + p] - knot[i]
            len2 = knot[i + p + 1] - knot[i + 1]
            if len1==0:
                c1 = 0
            else:
                c1 = (u - knot[i]) / (knot[i + p] - knot[i]) * self.Base2(u, i, p - 1, knot)
            if len2==0:
                c2 = 0
            else:
                c2 = (knot[i + p + 1] - u) / (knot[i + p + 1] - knot[i + 1]) * self.Base2(u, i + 1, p - 1, knot)
            res = c1 + c2
        return res
    
    
    def B_derivative(self, u, i, p, knot):
        self.tau = knot
        if p == 0:
            return 0.0
        c1, c2 = 0.0, 0.0
        
        if u == knot[-1]:# or u==knot[-1]:
            if knot[i + p] > knot[i]:
                c1 = p / (knot[i + p] - knot[i]) * self.Base2(u, i, p - 1, knot)
            if knot[i + p + 1] > knot[i + 1]:
                c2 = p / (knot[i + p + 1] - knot[i + 1]) * self.Base2(u, i + 1, p - 1, knot)
        else:
            if knot[i + p] > knot[i]:
                c1 = p / (knot[i + p] - knot[i]) * self.Base(u, i, p - 1, knot)
            if knot[i + p + 1] > knot[i + 1]:
                c2 = p / (knot[i + p + 1] - knot[i + 1]) * self.Base(u, i + 1, p - 1, knot)
#         print('c1',c1, 'c2',c2)
        return c1 - c2
    
    def B_scd_derivative(self, u, i, p, knot):
        # t = t[1:-1]
        self.tau = knot
        if p <= 1:
            return 0.0
        c1, c2, c3, c4 = 0.0, 0.0, 0.0, 0.0
        len1 = knot[i + p] - knot[i]
        len2 = knot[i + p + 1] - knot[i + 1]
        len3 = knot[i + p - 1] - knot[i]
        len4 = knot[i + p] - knot[i + 1]
        len5 = knot[i + p + 1] - knot[i + 2]

        if u == knot[0] or u == knot[-1]:
            if len1 > 0 and len3 > 0:
                c1 = p * (p - 1) / (len1 * len3) * self.Base2(u, i, p - 2, knot)
            if len1 > 0 and len4 > 0:
                c2 = p * (p - 1) / (len1 * len4) * self.Base2(u, i + 1, p - 2, knot)

            if len2 > 0 and len4 > 0:
                c3 = p * (p - 1) / (len2 * len4) * self.Base2(u, i + 1, p - 2, knot)
            if len2 > 0 and len5 > 0:
                c4 = p * (p - 1) / (len2 * len5) * self.Base2(u, i + 2, p - 2, knot)
        else:
            if len1 > 0 and len3 > 0:
                c1 = p * (p - 1) / (len1 * len3) * self.Base(u, i, p - 2, knot)
            if len1 > 0 and len4 > 0:
                c2 = p * (p - 1) / (len1 * len4) * self.Base(u, i + 1, p - 2, knot)

            if len2 > 0 and len4 > 0:
                c3 = p * (p - 1) / (len2 * len4) * self.Base(u, i + 1, p - 2, knot)
            if len2 > 0 and len5 > 0:
                c4 = p * (p - 1) / (len2 * len5) * self.Base(u, i + 2, p - 2, knot)
        return c1 - c2 - c3 + c4
    
    def build_spline_mat(self, data, K, p, knot):
        """
        Evaluate Pointwise Spline Matrix
        K:inner knot
        p:order
        k+p-1:basis function
        """
        #         print(data)
        B = np.zeros(shape=(data.shape[0], K + p))
        # print(data.shape[0])
        # print(K, p)
        for n in range(K + p):
            for i in range(B.shape[0]):
                B[i, n] = self.Base(data[i], n, p - 1, knot)
        B[-1, -1] = 1
        return B

    def build_deri_mat(self, data, K, p, knot):
        """
        derivative of data
        :param p: order
        :param data: t
        """
        D = np.zeros(shape=(data.shape[0], K + p))
        for n in range(K + p):
            for i in range(data.shape[0]):
                D[i, n] = self.B_derivative(data[i], n, p - 1, knot)
        #                 print('data[i]',i,data[i])
        #                 print('n',n)
        #         D[-1,-1] = 1
        return D

    def build_scd_deri_mat(self, data, K, p, knot):
        """
         2nd derivative of data
        :param p: order
        :param data: t
        """
        D2 = np.zeros(shape=(data.shape[0], K + p))
        for n in range(K + p):
            for i in range(data.shape[0]):
                D2[i, n] = self.B_scd_derivative(data[i], n, p - 1, knot)
        #         D2[-1,-1] = 1
        return D2
    
    def first_derivative(self, t, x, y, z, p, nvt, nvx, nvy, nvz):
        """
        1st derivative of t.
        :param p: order of bspline (degree+1),order = 4
        :param nvt, nvx, nvy, nvz: interior knots
        :param t: t, order:M1 = 3
        :param x: x, order:M2 = 3
        :param y: y, order:M3 = 3
        :param z: z, order:M4 = 3
        """
        K1 = len(nvt)
        tau1 = self.build_knot_vec(nvt, p, t)
        K2 = len(nvx)
        tau2 = self.build_knot_vec(nvx, p, x)
        K3 = len(nvy)
        tau3 = self.build_knot_vec(nvy, p, y)
        K4 = len(nvz)
        tau4 = self.build_knot_vec(nvz, p, z)
        
        D_t = self.build_deri_mat(t, K1, p, tau1)
        B_x = self.build_spline_mat(x, K2, p, tau2)
        B_y = self.build_spline_mat(y, K3, p, tau3)
        B_z = self.build_spline_mat(z, K4, p, tau4)

        X = np.kron(B_x, np.kron(B_y, np.kron(B_z, D_t)))
        
#         print('X_derivative', X)
#         print('X_derivative shape', X.shape)
        return X
    
    def second_derivative(self, t, x, y, z, p, nvt, nvx, nvy, nvz):
        """
        2nd derivative of xyz.
        :param p: order of bspline (degree+1),order = 4
        :param nvt, nvx, nvy, nvz: interior knots
        :param t: t, order:M1 = 3
        :param x: x, order:M2 = 3
        :param y: y, order:M3 = 3
        :param z: z, order:M4 = 3
        """
        K1 = len(nvt)
        tau1 = self.build_knot_vec(nvt, p, t)
        K2 = len(nvx)
        tau2 = self.build_knot_vec(nvx, p, x)
        K3 = len(nvy)
        tau3 = self.build_knot_vec(nvy, p, y)
        K4 = len(nvz)
        tau4 = self.build_knot_vec(nvz, p, z)
        
        B_t = self.build_spline_mat(t, K1, p, tau1)
        B_x = self.build_spline_mat(x, K2, p, tau2)
        B_y = self.build_spline_mat(y, K3, p, tau3)
        B_z = self.build_spline_mat(z, K4, p, tau4)
        
        d2_x = self.build_scd_deri_mat(x, K2, p, tau2)
        d2_y = self.build_scd_deri_mat(y, K3, p, tau3)
        d2_z = self.build_scd_deri_mat(z, K4, p, tau4)

        # D2_x1 = np.kron(np.kron(np.kron(B_x, B_y), d2_z), B_t)
        # D2_y1 = np.kron(np.kron(np.kron(B_x, d2_y), B_z), B_t)
        # D2_z1 = np.kron(np.kron(np.kron(d2_x, B_y), B_z), B_t)
        
        D2_x = np.kron(d2_x, np.kron(B_y, np.kron(B_z, B_t)))
        D2_y = np.kron(B_x, np.kron(d2_y, np.kron(B_z, B_t)))
        D2_z = np.kron(B_x, np.kron(B_y, np.kron(d2_z, B_t)))

        return D2_x, D2_y, D2_z
    
    def build_knot_vec(self, iknot, p, x):
        """
        build Open Uniform knot vector
        extension: duplicate knots
        """
        arr1 = np.array([x.min()] * (p))
        arr2 = np.array([x.max()] * (p))
        res = np.concatenate((arr1, iknot))
        res = np.concatenate((res, arr2))
        return  res

    def fit(self, x, y, p, iknot):
        """
        least squares of spline coefficients on data.
        :param p: order of bspline (degree+1)
        :param iknot: interior knots
        """

        tau = self.build_knot_vec(iknot, p, x)
        K = len(iknot)
        B = self.build_spline_mat(x, K, p, tau)#关于x的基函数矩阵，x决定行数，k+p-1为列数即基函数个数
        print('B:', B)
        print("Number of basis function：", B.shape[1])
        b = np.linalg.inv(B.T @ B) @ B.T @ y
        y_hat = B @ b
        mse = np.mean((y - y_hat) ** 2)   
        print('Training data fitting error：', mse)

        self.settings['tau'] = self.tau
        self.settings['p'] = p
        self.settings['K'] = K

        self.fitted['B'] = B
        self.fitted['b'] = b
        self.fitted['y_hat'] = y_hat
        self.fitted['mse'] = mse
        self.fitted['b_var_hat'] = np.linalg.inv(B.T @ B) * mse
    
    def fit4D(self, t, x, y, z, Y, p, nvt, nvx, nvy, nvz):
        """
        least squares of spline coefficients on data.
        :param p: order of bspline (degree+1),order = 4
        :param nvt, nvx, nvy, nvz: interior knots
        :param t: t, order:M1 = 3
        :param x: x, order:M2 = 3
        :param y: y, order:M3 = 3
        :param z: z, order:M4 = 3
        """
        K1 = len(nvt)
        tau1 = self.build_knot_vec(nvt, p, t)
        K2 = len(nvx)
        tau2 = self.build_knot_vec(nvx, p, x)
        K3 = len(nvy)
        tau3 = self.build_knot_vec(nvy, p, y)
        K4 = len(nvz)
        tau4 = self.build_knot_vec(nvz, p, z)

        B_t = self.build_spline_mat(t, K1, p, tau1)
        B_x = self.build_spline_mat(x, K2, p, tau2)
        B_y = self.build_spline_mat(y, K3, p, tau3)
        B_z = self.build_spline_mat(z, K4, p, tau4)
        print(B_t.shape)
        print(B_x.shape)
        print(B_y.shape)
        print(B_z.shape)
        B = np.kron(B_x, np.kron(B_y, np.kron(B_z, B_t)))

        b = np.linalg.inv(B.T @ B) @ B.T @ Y
        Y_hat = B @ b
        mse = np.mean((Y - Y_hat) ** 2)

        # print('b', len(b), b)
        # print('Y_hat', len(Y_hat), Y_hat)
        # print('MSE', mse)

        self.settings['tau1'] = tau1
        self.settings['tau2'] = tau2
        self.settings['tau3'] = tau3
        self.settings['tau4'] = tau4
        self.settings['p'] = p
        self.settings['K1'] = K1
        self.settings['K2'] = K2
        self.settings['K3'] = K3
        self.settings['K4'] = K4

        self.fitted['B'] = B
        self.fitted['b'] = b
        self.fitted['Y_hat'] = Y_hat
        self.fitted['mse'] = np.mean((Y - Y_hat) ** 2)
        self.fitted['b_var_hat'] = np.linalg.inv(B.T @ B) * mse
    
    def predict(self, x):
        h_x = self.build_spline_mat(x, self.settings['K'], self.settings['p'], self.settings['tau'])
        return h_x @ self.fitted['b']
    
    def predict_4D(self, t, x, y, z):
        h_t = self.build_spline_mat(t, self.settings['K1'], self.settings['p'], self.settings['tau1'])
        h_x = self.build_spline_mat(x, self.settings['K2'], self.settings['p'], self.settings['tau2'])
        h_y = self.build_spline_mat(y, self.settings['K3'], self.settings['p'], self.settings['tau3'])
        h_z = self.build_spline_mat(z, self.settings['K4'], self.settings['p'], self.settings['tau4'])
        h = np.kron(h_x, np.kron(h_y, np.kron(h_z, h_t)))
        return h@ self.fitted['b']
    
    def predict_B(self, t, x, y, z):
        h_t = self.build_spline_mat(t, self.settings['K1'], self.settings['p'], self.settings['tau1'])
        h_x = self.build_spline_mat(x, self.settings['K2'], self.settings['p'], self.settings['tau2'])
        h_y = self.build_spline_mat(y, self.settings['K3'], self.settings['p'], self.settings['tau3'])
        h_z = self.build_spline_mat(z, self.settings['K4'], self.settings['p'], self.settings['tau4'])
        h = np.kron(h_x, np.kron(h_y, np.kron(h_z, h_t)))
        return h
    
    def beta(self):
        return self.fitted['b']

    def Y_hat(self):
        return self.fitted['Y_hat']

    def get_B(self):
        return self.fitted['B']

df1 = np.load("input/xg-hum1030-l1.npy")
df2 = np.load("input/xg-hum1030-l2.npy")
df3 = np.load("input/xg-hum1030-l3.npy")
t_total = 1769
boundary = np.zeros([3, 6, 3, t_total])
for t in range(t_total):
    for i in range(3):
        for j in range(6):
            boundary[i][j][0][t] = df1[t][6 * i + j + 1]
            boundary[i][j][1][t] = df2[t][6 * i + j + 1]
            boundary[i][j][2][t] = df3[t][6 * i + j + 1]

if __name__ == '__main__':
    tp = 31#31
    xp = 3
    yp = 6
    zp = 3
    
    # domain
    Max_x = 9.8
    Max_y = 24.5
    Max_z = 3.8
    Max_t = 720
    rhot = int(Max_t/(tp-1))
    
    #  node vector
    nodeVector_x = []  # M = 3
    nodeVector_y = [Max_y/(yp-2)*(i+1) for i in range(yp-3)]  # M = 3
    nodeVector_z = []  # M = 3

    nodeVector_t = [Max_t/(tp-2)*(i+1) for i in range(tp-3)]    # M = 4

    # t, x, y, z
    t = np.linspace(0, Max_t, tp, endpoint=True)
    x = np.linspace(0, Max_x, xp, endpoint=True)
    y = np.linspace(0, Max_y, yp, endpoint=True)
    z = np.linspace(0, Max_z, zp, endpoint=True)

    # grid data
    grid_4d = boundary[:, :, :, :(Max_t+1):rhot]
    print(grid_4d.shape)
    
    # vector data
    hum_4d = np.zeros(tp * zp * yp * xp)
    for l in range(tp):
        for k in range(zp):
            for j in range(yp):
                for i in range(xp):
                    hum_4d[tp * (zp * (yp * i + j) + k) + l] = grid_4d[i, j, k, l]
#     print(grid_4d)
    
    import random
    
    # Bspline model
    bsp_4d = Bspline()
    bsp_4d.fit4D(t, x, y, z, hum_4d, 3, nodeVector_t, nodeVector_x, nodeVector_y, nodeVector_z)
    beta_4 = bsp_4d.beta()
    # print('beta_4', beta_4)
    print('beta_4 shape(K):', beta_4.shape)

# testing data
new_tp = 91 #8*90
rhot_new = 8
tt = np.linspace(0, Max_t, new_tp)
# hum_estimation:1-dim
hum_estimation = bsp_4d.predict_4D(tt, x, y, z)
print('n:', hum_estimation.shape)
hum_pre_restore = np.zeros([xp, yp, zp, new_tp])#unknown time point
# hum_pre_restore: 4-dim
for i in range(xp):
    for j in range(yp):
        for k in range(zp):
            for l in range(new_tp):
                hum_pre_restore[i, j, k, l]= hum_estimation[new_tp * (zp * (yp * i + j) + k) + l]

# validation_origin4-dim,hum_validation:1-dim
validation_origin = boundary[:, :, :, :(Max_t+1):rhot_new]
hum_validation = np.zeros(xp*yp*zp*new_tp)
for l in range(new_tp):
        for k in range(zp):
            for j in range(yp):
                for i in range(xp):
                    hum_validation[new_tp * (zp * (yp * i + j) + k) + l] = validation_origin[i, j, k, l]

#Bspine error
error_bs1 = hum_estimation - hum_validation
error_ss = error_bs1.T@error_bs1

print('mean_BSpline', np.mean(error_bs1))
print('mean_abs_BSpline', np.mean(np.abs(error_bs1)))
print('max_BSpline', np.max(error_bs1))
print('min_BSpline', np.min(error_bs1))
print('rmse_BSpline', np.sqrt(error_ss/error_bs1.shape[0]))
print('SST_BSpline', error_ss)

# plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['font.family'] = ['Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
font = {'family': 'Times New Roman',
        'style': 'normal',
        'weight': 'normal',
        'color': 'black',
        'size': 25}

# time variation figure 6(point:integer, no bigger than xp/yp/zp)
import datetime

now = datetime.datetime.now()

xtick = np.linspace(0, Max_t, 12+1, endpoint=True)
fig1, ax1 = plt.subplots(1,1,figsize=(9,6))

x1_point = 0
y1_point = 3
z1_point = 2

x2_point = 1
y2_point = 3
z2_point = 2

x3_point = 2
y3_point = 3
z3_point = 2

x4_point = 1
y4_point = 1
z4_point = 1

x5_point = 1
y5_point = 2
z5_point = 1

x6_point = 1
y6_point = 3
z6_point = 1

ax1.plot(tt, boundary[x1_point, y1_point, z1_point, 0:(Max_t+1):rhot_new], '4', color='#ad0afd', label=r'(%.1f, %.1f, %.1f)'%(x1_point*Max_x/(xp-1), y1_point*Max_y/(yp-1), z1_point*Max_z/(zp-1)), markersize=6, zorder=3, clip_on=False)
ax1.plot(tt, boundary[x2_point, y2_point, z2_point, 0:(Max_t+1):rhot_new], 'x',  color='#ef4026', label=r'(%.1f, %.1f, %.1f)'%(x2_point*Max_x/(xp-1), y2_point*Max_y/(yp-1), z2_point*Max_z/(zp-1)), markersize=6, zorder=3, clip_on=False)
ax1.plot(tt, boundary[x3_point, y3_point, z3_point, 0:(Max_t+1):rhot_new], '*',  color='#2976bb', label=r'(%.1f, %.1f, %.1f)'%(x3_point*Max_x/(xp-1), y3_point*Max_y/(yp-1), z3_point*Max_z/(zp-1)), markersize=6, zorder=3, clip_on=False)

ax1.plot(tt, boundary[x4_point, y4_point, z4_point, 0:(Max_t+1):rhot_new], '+', color='#341c02', label=r'(%.1f, %.1f, %.1f)'%(x4_point*Max_x/(xp-1), y4_point*Max_y/(yp-1), z4_point*Max_z/(zp-1)), markersize=6, zorder=3, clip_on=False)
ax1.plot(tt, boundary[x5_point, y5_point, z5_point, 0:(Max_t+1):rhot_new], '.',  color='#009337', label=r'(%.1f, %.1f, %.1f)'%(x5_point*Max_x/(xp-1), y5_point*Max_y/(yp-1), z5_point*Max_z/(zp-1)), markersize=6, zorder=3, clip_on=False)
ax1.plot(tt, boundary[x6_point, y6_point, z6_point, 0:(Max_t+1):rhot_new], '<', color='#f9bc08', label=r'(%.1f, %.1f, %.1f)'%(x6_point*Max_x/(xp-1), y6_point*Max_y/(yp-1), z6_point*Max_z/(zp-1)), markersize=6, zorder=3, clip_on=False)

ax1.set_xlim(0, 720)
ax1.set_ylim(40, 46)
ax1.set_xlabel('Time/Hour', fontdict=font)
ax1.set_ylabel('Humidity/%', fontdict=font)

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
                 labelsize=20)

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

plt.legend(ncol=2, loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig('pic/Humidity_Profile_%s'%now.strftime('%Y%m%d'), dpi=1200)

T_f = bsp_4d.first_derivative(tt, x, y, z, 3, nodeVector_t, nodeVector_x, nodeVector_y, nodeVector_z)

X_d, Y_d, Z_d = bsp_4d.second_derivative(tt, x, y, z, 3, nodeVector_t, nodeVector_x, nodeVector_y, nodeVector_z)

zeta_ob = 1*T_f@beta_4 - 3600*1.37/10**8*(X_d@beta_4 +Y_d@beta_4 + Z_d@beta_4)
# print(zeta_ob)
print('mean_zeta', np.mean(zeta_ob))
print('mean_abs_zeta', np.mean(np.abs(zeta_ob)))
print('max_zeta', np.max(zeta_ob))
print('min_zeta', np.min(zeta_ob))
print('RMSE', np.sqrt(zeta_ob.T@zeta_ob/zeta_ob.shape[0]))
print('SST', zeta_ob.T@zeta_ob)

B_pre = bsp_4d.predict_B(tt, x, y, z)

tnew = np.linspace(0, Max_t, 2)
xx = np.linspace(0, Max_x, 100)
yy = np.linspace(0, Max_y, 100)
zz = np.linspace(0, Max_z, 79)
hum_pre_new = bsp_4d.predict_4D(tnew, xx, yy, zz)

restore2 = np.zeros([100, 100, 79, 2])
for i in range(100):
    for j in range(100):
        for k in range(79):
            for l in range(2):
                restore2[i, j, k, l]= hum_pre_new[2 * (79 * (100 * i + j) + k) + l]

maxt = np.max(restore2)
mint = np.min(restore2)
print(maxt, mint)

import matplotlib.ticker
import matplotlib.pyplot as plt
import random
X, Y, Z = np.meshgrid(xx, yy, zz)

v = restore2[:, :, :, 1]

min_v = np.min(v)
max_v = np.max(v)
print(min_v)

v_1d = v.reshape(790000,1)
color = [plt.get_cmap("cool", 100)(int(float(i-min_v)/(max_v-min_v)*100)) for i in v_1d]

fig = plt.figure()

ax = fig.add_subplot(111,projection='3d')
ax.grid(None)

ax.axis('off')

plt.set_cmap(plt.get_cmap("cool", 100))

im = ax.scatter(X, Y, Z, s=1,c=color,marker='o',linewidths=0)

fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x,pos:int(x*(max_v-min_v)+min_v)),label='Humidity/%')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_box_aspect([9.8, 24.5, 3.8])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

plt.savefig('pic/cool-hum-720.png', dpi=1200)
plt.show()