import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, gamma, norm, multivariate_normal, pearsonr
import copy
import time
import openpyxl

I = np.eye(3)
D = np.diff(I,2,axis=0)

#Definition of Bspline
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
        self.tau = knot  # tau 为inner knots加上左右两端点的重复knot
        if p == 0:
            if (knot[i] <= u) and (u < knot[i + 1]):
                res = 1
            else:
                res = 0
        else:
            c1, c2 = 0, 0
            len1 = knot[i + p] - knot[i]
            len2 = knot[i + p + 1] - knot[i + 1]
            if len1 == 0:
                c1 = 0
            else:
                c1 = (u - knot[i]) / (knot[i + p] - knot[i]) * self.Base(u, i, p - 1, knot)
            if len2 == 0:
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
        self.tau = knot  # tau 为inner knots加上左右两端点的重复knot
        if p == 0:
            if (knot[i] <= u) and (u <= knot[i + 1]):
                res = 1
            else:
                res = 0
        else:
            c1, c2 = 0, 0
            len1 = knot[i + p] - knot[i]
            len2 = knot[i + p + 1] - knot[i + 1]
            if len1 == 0:
                c1 = 0
            else:
                c1 = (u - knot[i]) / (knot[i + p] - knot[i]) * self.Base2(u, i, p - 1, knot)
            if len2 == 0:
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

        if u == knot[-1]:
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
        K为inter knot数
        p是order
        k+p-1是基函数个数
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

    def first_derivative_t(self, t, x, p, nvt, nvx):
        """
        1st derivative of t.
        :param p: order of bspline (degree+1),order = 5
        :param nvt, nvx: interior knots
        :param t: t, order:M1 = 5
        :param x: x, order:M2 = 5
        """
        K1 = len(nvt)
        tau1 = self.build_knot_vec(nvt, p, t)
        # print(tau1)
        K2 = len(nvx)
        tau2 = self.build_knot_vec(nvx, p, x)

        D_t = self.build_deri_mat(t, K1, p, tau1)
        # print(D_t)
        B_x = self.build_spline_mat(x, K2, p, tau2)

        X = np.kron(B_x, D_t)
        #         print(X)
        return X

    def second_derivative_x(self, t, x, p, nvt, nvx):
        """
        2nd derivative of x.
        :param p: order of bspline (degree+1),order = 5
        :param nvt, nvx: interior knots
        :param t: t, order:M1 = 5
        :param x: x, order:M2 = 5
        """
        K1 = len(nvt)
        tau1 = self.build_knot_vec(nvt, p, t)
        K2 = len(nvx)
        tau2 = self.build_knot_vec(nvx, p, x)

        B_t = self.build_spline_mat(t, K1, p, tau1)
        d2_x = self.build_scd_deri_mat(x, K2, p, tau2)

        X = np.kron(d2_x, B_t)
        return X

    def first_derivative_x(self, t, x, p, nvt, nvx):
        """
        1st derivative of x.
        :param p: order of bspline (degree+1),order = 5
        :param nvt, nvx: interior knots
        :param t: t, order:M1 = 5
        :param x: x, order:M2 = 5
        """
        K1 = len(nvt)
        tau1 = self.build_knot_vec(nvt, p, t)
        K2 = len(nvx)
        tau2 = self.build_knot_vec(nvx, p, x)

        B_t = self.build_spline_mat(t, K1, p, tau1)
        D_x = self.build_deri_mat(x, K2, p, tau2)
        #         print(D_x)
        X = np.kron(D_x, B_t)
        #         print(X)
        return X

    def build_knot_vec(self, iknot, p, x):
        """
        build Open Uniform knot vector
        extension: duplicate knots
        """
        arr1 = np.array([x.min()] * (p))
        arr2 = np.array([x.max()] * (p))
        res = np.concatenate((arr1, iknot))
        res = np.concatenate((res, arr2))
        return res

    def fit(self, x, y, p, iknot):
        """
        least squares of spline coefficients on data.
        :param p: order of bspline (degree+1)
        :param iknot: interior knots
        """

        tau = self.build_knot_vec(iknot, p, x)
        # print(tau)
        # print(x)
        K = len(iknot)
        B = self.build_spline_mat(x, K, p, tau)  # 关于x的基函数矩阵，x决定行数，k+p为列数即基函数个数
        # print('B:', B)
        print("基函数个数为：", B.shape[1])
        b = np.linalg.inv(B.T @ B) @ B.T @ y
        y_hat = B @ b
        mse = np.mean((y - y_hat) ** 2)
        print('训练样本拟合均方误差为：', mse)

        self.settings['tau'] = self.tau
        self.settings['p'] = p
        self.settings['K'] = K

        self.fitted['B'] = B
        self.fitted['b'] = b
        self.fitted['y_hat'] = y_hat
        self.fitted['mse'] = mse
        self.fitted['b_var_hat'] = np.linalg.inv(B.T @ B) * mse

    def fit2D(self, t, x, Y, p, nvt, nvx, Lambda1, Lambda2, penalty_order):
        """
        least squares of spline coefficients on data.
        :param p: order of bspline (degree+1),order = 5
        :param nvt, nvx: interior knots
        :param t: t, order:M1 = 5
        :param x: x, order:M2 = 5
        """
        K1 = len(nvt)
        tau1 = self.build_knot_vec(nvt, p, t)
        K2 = len(nvx)
        tau2 = self.build_knot_vec(nvx, p, x)

        B_t = self.build_spline_mat(t, K1, p, tau1)
        B_x = self.build_spline_mat(x, K2, p, tau2)
        print(B_t.shape)
        print(B_x.shape)

        I1 = np.eye(B_x.shape[1])
        I2 = np.eye(B_t.shape[1])

        d_t = np.diff(I2, penalty_order, axis=0)
        d_x = np.diff(I1, penalty_order, axis=0)
        
        H1 = np.kron(B_x.T @ B_x, d_t.T @ d_t)
        H2 = np.kron(d_x.T @ d_x, B_t.T @ B_t)
        # H3 = np.kron(d_x.T @ d_x, d_t.T @ d_t)

        P1 = np.kron(I1, d_t.T @ d_t)
        P2 = np.kron(d_x.T @ d_x, I2)
        P3 = np.kron(d_x.T @ d_x, d_t.T @ d_t)
        
        # lam_sqrt = np.sqrt(Lambda1)
        
        B = np.kron(B_x, B_t)
        # p_matrix1 = lam_sqrt*np.kron(d_x, I2)
        # p_matrix2 = lam_sqrt*np.kron(I1, d_t)
        # # p_matrix3 = lam_sqrt*np.kron(d_x, d_t)
        # p_matrix = lam_sqrt*np.kron(d_x, d_t)
        
        # B_mid1 = np.concatenate((B, p_matrix1), axis=0)
        # B_mid2 = np.concatenate((B_mid1, p_matrix2), axis=0)
        # B_aug = B_mid2
        # B_aug = np.concatenate((B, p_matrix), axis=0)
        
        # Y_mid = np.zeros(p_matrix1.shape[0] + p_matrix2.shape[0])
        # Y_mid = np.zeros(p_matrix.shape[0])
        # Y_aug = np.concatenate((Y, Y_mid), axis=0)
        # b = np.linalg.inv(B.T @ B) @ B.T @ Y
        # b = np.linalg.inv(B_aug.T @ B_aug) @ B_aug.T @ Y_aug
        b = np.linalg.inv(B.T @ B + Lambda1*P1+Lambda2*P2) @ B.T @ Y
        Y_hat = B @ b
        mse = np.mean((Y - Y_hat) ** 2)

        print('b', len(b), b)
        print('Y_hat', len(Y_hat), Y_hat)
        print('MSE', mse)

        self.settings['tau1'] = tau1
        self.settings['tau2'] = tau2
        self.settings['p'] = p
        self.settings['K1'] = K1
        self.settings['K2'] = K2

        self.fitted['B'] = B
        self.fitted['H1'] = H1
        self.fitted['H2'] = H2
        self.fitted['P1'] = P1
        self.fitted['P2'] = P2
        self.fitted['P3'] = P3
        self.fitted['b'] = b
        self.fitted['Y_hat'] = Y_hat
        self.fitted['mse'] = np.mean((Y - Y_hat) ** 2)
        # plt.plot(Y - Y_hat)
        print('统计模型误差最大值：', np.max(Y - Y_hat))
        print('统计模型误差最小值：', np.min(Y - Y_hat))
        self.fitted['b_var_hat'] = np.linalg.inv(B.T @ B) * mse

    #     def fit4D(self, t, x, y, z, Y, p, nvt, nvx, nvy, nvz):
    #         """
    #         least squares of spline coefficients on data.
    #         :param p: order of bspline (degree+1),order = 4
    #         :param nvt, nvx, nvy, nvz: interior knots
    #         :param t: t, order:M1 = 5
    #         :param x: x, order:M2 = 4
    #         :param y: y, order:M3 = 5
    #         :param z: z, order:M4 = 4
    #         """
    #         K1 = len(nvt)
    #         tau1 = self.build_knot_vec(nvt, p + 1, t)
    #         K2 = len(nvx)
    #         tau2 = self.build_knot_vec(nvx, p, x)
    #         K3 = len(nvy)
    #         tau3 = self.build_knot_vec(nvy, p + 1, y)
    #         K4 = len(nvz)
    #         tau4 = self.build_knot_vec(nvz, p, z)

    #         B_t = self.build_spline_mat(t, K1, p+1, tau1)
    #         B_x = self.build_spline_mat(x, K2, p, tau2)
    #         B_y = self.build_spline_mat(y, K3, p+1, tau3)
    #         B_z = self.build_spline_mat(z, K4, p, tau4)
    #         print(B_t.shape)
    #         print(B_x.shape)
    #         print(B_y.shape)
    #         print(B_z.shape)
    #         B = np.kron(np.kron(np.kron(B_z, B_y), B_x), B_t)

    #         b = np.linalg.inv(B.T @ B) @ B.T @ Y
    #         Y_hat = B @ b
    #         mse = np.mean((Y - Y_hat) ** 2)

    #         print('b', len(b), b)
    #         print('Y_hat', len(Y_hat), Y_hat)
    #         print('MSE', mse)

    #         self.settings['tau1'] = tau1
    #         self.settings['tau2'] = tau2
    #         self.settings['tau3'] = tau3
    #         self.settings['tau4'] = tau4
    #         self.settings['p'] = p
    #         self.settings['K1'] = K1
    #         self.settings['K2'] = K2
    #         self.settings['K3'] = K3
    #         self.settings['K4'] = K4

    #         self.fitted['B'] = B
    #         self.fitted['b'] = b
    #         self.fitted['Y_hat'] = Y_hat
    #         self.fitted['mse'] = np.mean((Y - Y_hat) ** 2)
    #         self.fitted['b_var_hat'] = np.linalg.inv(B.T @ B) * mse

    def predict(self, x):
        h_x = self.build_spline_mat(x, self.settings['K'], self.settings['p'], self.settings['tau'])
        return h_x @ self.fitted['b']

    def predict_2D(self, t, x):
        h_t = self.build_spline_mat(t, self.settings['K1'], self.settings['p'], self.settings['tau1'])
        h_x = self.build_spline_mat(x, self.settings['K2'], self.settings['p'], self.settings['tau2'])
        h = np.kron(h_x, h_t)
        return h @ self.fitted['b']

    #     def predict_4D(self, t, x, y, z):
    #         h_t = self.build_spline_mat(t, self.settings['K1'], self.settings['p'] + 1, self.settings['tau1'])
    #         h_x = self.build_spline_mat(x, self.settings['K2'], self.settings['p'], self.settings['tau2'])
    #         h_y = self.build_spline_mat(y, self.settings['K3'], self.settings['p'] + 1, self.settings['tau3'])
    #         h_z = self.build_spline_mat(z, self.settings['K4'], self.settings['p'], self.settings['tau4'])
    #         h = np.kron(np.kron(np.kron(h_z, h_y), h_x), h_t)
    #         return h@ self.fitted['b']

    #     def predict_B(self, t, x, y, z):
    #         h_t = self.build_spline_mat(t, self.settings['K1'], self.settings['p'] + 1, self.settings['tau1'])
    #         h_x = self.build_spline_mat(x, self.settings['K2'], self.settings['p'], self.settings['tau2'])
    #         h_y = self.build_spline_mat(y, self.settings['K3'], self.settings['p'] + 1, self.settings['tau3'])
    #         h_z = self.build_spline_mat(z, self.settings['K4'], self.settings['p'], self.settings['tau4'])
    #         h = np.kron(np.kron(np.kron(h_z, h_y), h_x), h_t)
    #         return h

    def predict_B_2D(self, t, x):
        h_t = self.build_spline_mat(t, self.settings['K1'], self.settings['p'], self.settings['tau1'])
        h_x = self.build_spline_mat(x, self.settings['K2'], self.settings['p'], self.settings['tau2'])

        h = np.kron(h_x, h_t)
        return h

    def penalty_2D(self, t, x, order_p):
        b_t = self.build_spline_mat(t, self.settings['K1'], self.settings['p'], self.settings['tau1'])
        b_x = self.build_spline_mat(x, self.settings['K2'], self.settings['p'], self.settings['tau2'])

        I1 = np.eye(b_x.shape[1])
        I2 = np.eye(b_t.shape[1])

        d_t = np.diff(I2, order_p, axis=0)
        d_x = np.diff(I1, order_p, axis=0)
        H1 = np.kron(b_x.T@b_x, d_t.T@d_t)
        H2 = np.kron(d_x.T@d_x, b_t.T@b_t)
        H3 = np.kron(d_x.T@d_x, d_t.T@d_t)
        return H1, H2, H3

    def predict_B(self, t):
        h_t = self.build_spline_mat(t, self.settings['K1'], self.settings['p'], self.settings['tau1'])
        return h_t

    def predict_theta(self, t):
        h_t = self.build_spline_mat(t, self.settings['K'], self.settings['p'], self.settings['tau'])
        return h_t

    def beta(self):
        return self.fitted['b']

    def Y_hat(self):
        return self.fitted['Y_hat']

    def get_B(self):
        return self.fitted['B']

order = 4
order_psi = 3

# kt&kx,t/x num of basis
tp = 17
xp = 11 - 2
nt = 41
nx = 33

sigma_noise = 0.01

# smooth
penalty_lambda1 = 0#0.1 1e-2
penalty_lambda2 = 0#0.001 1e-6 
order_p = 3

# mean value
thetax1_node = np.array([0.0431, 0.0478, 0.0512, 0.0546, 0.0587])
thetax2_node = np.array([0.1021, 0.0998, 0.1013, 0.1025, 0.1054])
thetax3_node = np.array([0.1105, 0.1083, 0.1094, 0.1112, 0.1098])

num_node = thetax1_node.shape[0] #time node
num_fun = 3
# cov matrix
S1 = np.array([[0.0000015, 0.0000012, 0.0000012], [0.0000012, 0.0000015, 0.0000012], [0.0000012, 0.0000012, 0.0000015]])

# generation
alpha = np.zeros([num_node, 3, 3])
for i in range(num_node):
    alpha[i][0] = np.random.multivariate_normal([thetax1_node[i] for j in range(num_fun)], S1)
    alpha[i][1] = np.random.multivariate_normal([thetax2_node[i] for j in range(num_fun)], S1)
    alpha[i][2] = np.random.multivariate_normal([thetax3_node[i] for j in range(num_fun)], S1)
alpha = np.around(alpha, decimals=6)
print('Time-Varying parameters:\n', alpha.T)

# Initialization
delta_z = 0.025 #h spatial grid
delta_t = 0.005 #k time grid

Max_z = 10 #l 
Max_t = 2

n = int(Max_z/delta_z) + 1
m = int(Max_t/delta_t) + 1

U1 = np.zeros([n, m])  #(z, t)
U2 = np.zeros([n, m])  #(z, t)
U3 = np.zeros([n, m])  #(z, t)

print('Nz:', n, 'Nt:', m)

# boundary condition
def f_1(t):
    temp = 0
    return temp

def f_2(t):
    temp = 0
    return temp
# initial condition
def phi1(z):
    temp = 1/(1 + 0.1 * (Max_z/2 - z)**2)
    return temp

def phi2(z):
    temp = 1/(1 + 0.1 * (Max_z/2 - z)**2)
    return temp

def phi3(z):
    temp = 1/(1 + 0.1 * (Max_z/2 - z)**2)
    return temp

# Bspline knot sequence
knot_t0 = [Max_t/(num_node-2)*(i+1) for i in range(num_node-order_psi)]
t_theta = np.linspace(0, Max_t, num_node)

# time-vatying parameter of Bspline
theta11_bs = Bspline()
theta11_bs.fit(t_theta, (alpha.T)[0][0], order_psi, knot_t0)

theta12_bs = Bspline()
theta12_bs.fit(t_theta, (alpha.T)[0][1], order_psi, knot_t0)

theta13_bs = Bspline()
theta13_bs.fit(t_theta, (alpha.T)[0][2], order_psi, knot_t0)

theta21_bs = Bspline()
theta21_bs.fit(t_theta, (alpha.T)[1][0], order_psi, knot_t0)

theta22_bs = Bspline()
theta22_bs.fit(t_theta, (alpha.T)[1][1], order_psi, knot_t0)

theta23_bs = Bspline()
theta23_bs.fit(t_theta, (alpha.T)[1][2], order_psi, knot_t0)

theta31_bs = Bspline()
theta31_bs.fit(t_theta, (alpha.T)[2][0], order_psi, knot_t0)

theta32_bs = Bspline()
theta32_bs.fit(t_theta, (alpha.T)[2][1], order_psi, knot_t0)

theta33_bs = Bspline()
theta33_bs.fit(t_theta, (alpha.T)[2][2], order_psi, knot_t0)

# parameter density
t_pre = np.linspace(0, Max_t, m)
theta_11 = theta11_bs.predict(t_pre)
theta_12 = theta12_bs.predict(t_pre)
theta_13 = theta13_bs.predict(t_pre)
theta_21 = theta21_bs.predict(t_pre)
theta_22 = theta22_bs.predict(t_pre)
theta_23 = theta23_bs.predict(t_pre)
theta_31 = theta31_bs.predict(t_pre)
theta_32 = theta32_bs.predict(t_pre)
theta_33 = theta33_bs.predict(t_pre)

# plt.plot(theta_11[:], label=r'$\psi_{11}$')
# plt.plot(theta_12[:], label=r'$\psi_{12}$')
# plt.plot(theta_13[:], label=r'$\psi_{13}$')
# plt.legend()

for j in range(m): # boundary condition 0
    U1[0, j] = f_1(j*delta_t)
    U1[n-1, j] = f_2(j*delta_t)
    U2[0, j] = f_1(j*delta_t)
    U2[n-1, j] = f_2(j*delta_t)
    U3[0, j] = f_1(j*delta_t)
    U3[n-1, j] = f_2(j*delta_t)
    
for i in range(n):  # initial condition（z, t）
    U1[i, 0] = phi1(i * delta_z)
    U2[i, 0] = phi2(i * delta_z)
    U3[i, 0] = phi3(i * delta_z)

def index11(t):
    index1 = 1 + delta_t*(-2*theta_11[t]/(delta_z**2) + theta_12[t]/delta_z + theta_13[t] )
    return index1

def index12(t):
    index2 = delta_t*(theta_11[t]/(delta_z**2))
    return index2

def index13(t):
    index3 = delta_t*(theta_11[t]/delta_z**2- theta_12[t]/delta_z)
    return index3


U1[1, 1] = index11(0) * U1[1, 0]  + index12(0) * U1[2, 0] + index13(0) * U1[0, 0]
for i in range(1, n-1):
    U1[i, 1] = index11(0) * U1[i, 0]  + index12(0) * U1[i+1, 0] + index13(0) * U1[i-1, 0]

for j in range(1, m-1):
    U1[1, j+1] = index11(j) * U1[1, j]  + index12(j) * U1[1+1, j] + index13(j) * U1[1, j]
    U1[n-2, j+1] = index11(j) * U1[n-2, j]  + index12(j) * U1[n-2, j] + index13(j) * U1[n-3, j]

    for i in range(2, n-2):
        U1[i, j+1] = index11(j) * U1[i, j]  + index12(j) * U1[i+1, j] + index13(j) * U1[i-1, j]
for j in range(1, m):#z boundary
    U1[0, j] = index11(j-1) * U1[0, j-1]  + index12(j-1) * U1[1, j-1] + index13(j-1) * U1[0, j-1]
    U1[n-1, j] = index11(j-1) * U1[n-1, j-1]  + index12(j-1) * U1[n-1, j-1] + index13(j-1) * U1[n-2, j-1]

# finite difference
def index21(t):
    index1 = 1 + delta_t*(-2*theta_21[t]/(delta_z**2) + theta_22[t]/delta_z + theta_23[t])
    return index1

def index22(t):
    index2 = delta_t*(theta_21[t]/(delta_z**2))
    return index2

def index23(t):
    index3 = delta_t*(theta_21[t]/delta_z**2 - theta_22[t]/delta_z)
    return index3

for i in range(1, n-1):
    U2[i, 1] = index21(0) * U2[i, 0]  + index22(0) * U2[i+1, 0] + index23(0) * U2[i-1, 0]
        
for j in range(1, m-1):
    U2[1, j+1] = index21(j) * U2[1, j]  + index22(j) * U2[1+1, j] + index23(j) * U2[1, j]
    U2[n-2, j+1] = index21(j) * U2[n-2, j]  + index22(j) * U2[n-2, j] + index23(j) * U2[n-3, j]
    
    for i in range(2, n-2):
        U2[i, j+1] = index21(j) * U2[i, j]  + index22(j) * U2[i+1, j] + index23(j) * U2[i-1, j]
for j in range(1, m):#z边界
    U2[0, j] = index21(j-1) * U2[0, j-1]  + index22(j-1) * U2[1, j-1] + index23(j-1) * U2[0, j-1]
    U2[n-1, j] = index21(j-1) * U2[n-1, j-1]  + index22(j-1) * U2[n-1, j-1] + index23(j-1) * U2[n-2, j-1]

def index31(t):
    index1 = 1 + delta_t*(-2*theta_31[t]/(delta_z**2) + theta_32[t]/delta_z + theta_33[t])
    return index1

def index32(t):
    index2 = delta_t*(theta_31[t]/delta_z**2)
    return index2

def index33(t):
    index3 = delta_t*(theta_31[t]/delta_z**2 - theta_32[t]/delta_z)
    return index3

for i in range(1, n-1):
    U3[i, 1] = index31(0) * U3[i, 0]  + index32(0) * U3[i+1, 0] + index33(0) * U3[i-1, 0]
        
for j in range(1, m-1):
    U3[1, j+1] = index31(j) * U3[1, j]  + index32(j) * U3[1+1, j] + index33(j) * U3[1, j]
    U3[n-2, j+1] = index31(j) * U3[n-2, j]  + index32(j) * U3[n-2, j] + index33(j) * U3[n-3, j]
    
    for i in range(2, n-2):
        U3[i, j+1] = index31(j) * U3[i, j]  + index32(j) * U3[i+1, j] + index33(j) * U3[i-1, j]
for j in range(1, m):#z边界
    U3[0, j] = index31(j-1) * U3[0, j-1]  + index32(j-1) * U3[1, j-1] + index33(j-1) * U3[0, j-1]
    U3[n-1, j] = index31(j-1) * U3[n-1, j-1]  + index32(j-1) * U3[n-1, j-1] + index33(j-1) * U3[n-2, j-1]


# 3D plot (Figure 3)
paras = {
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
}
plt.rcParams.update(paras)

l_color = [
    '#EF476F', 
    '#FFD166', 
    '#06D6A0', 
    '#118AB2',
]
xo = np.linspace(0, Max_z, n)
yo = np.linspace(0, Max_t, m)

Xo, Yo = np.meshgrid(xo, yo)

fig3d= plt.figure(figsize=(15,6))
ax3d0 = fig3d.add_subplot(131, projection="3d")
ax3d0.set_xlim3d(0, Max_z)
ax3d0.set_ylim3d(0, Max_t)
# ax3d0.set_zlim3d(0, 8)
# ax3d0.xaxis.set_rotate_label(True)
# ax3d0.yaxis.set_rotate_label(True)
# ax3d0.zaxis.set_rotate_label(True)
ax3d0.set_xlabel("Z")
ax3d0.set_ylabel("T")
ax3d0.set_zlabel("PDE 1")
ax3d0.plot_surface(Xo, Yo, U1.T, cmap="plasma", linewidths=0.1, label='PDE 1')

ax3d0.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax3d0.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax3d0.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

plt.title('PDE 1')

ax3d1 = fig3d.add_subplot(132, projection="3d")
ax3d1.set_xlim3d(0, Max_z)
ax3d1.set_ylim3d(0, Max_t)
# ax3d1.set_zlim3d(0, 8)
# ax3d1.xaxis.set_rotate_label(True)
# ax3d1.yaxis.set_rotate_label(True)
# ax3d1.zaxis.set_rotate_label(True)
ax3d1.set_xlabel("Z")
ax3d1.set_ylabel("T")
ax3d1.set_zlabel("PDE 2")
ax3d1.plot_surface(Xo, Yo, U2.T, cmap="viridis", linewidths=0.1, label='PDE 2')

ax3d1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax3d1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax3d1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.title('PDE 2')

ax3d2 = fig3d.add_subplot(133, projection="3d")
ax3d2.set_xlim3d(0, Max_z)
ax3d2.set_ylim3d(0, Max_t)
# ax3d2.set_zlim3d(0, 8)
# ax3d2.xaxis.set_rotate_label(True)
# ax3d2.yaxis.set_rotate_label(True)
# ax3d2.zaxis.set_rotate_label(True)
ax3d2.set_xlabel("Z")
ax3d2.set_ylabel("T")
ax3d2.set_zlabel("PDE 3")
ax3d2.plot_surface(Xo, Yo, U3.T, cmap="summer", linewidths=0.1, label='PDE 3')

ax3d2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax3d2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax3d2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.title('PDE 3')
plt.savefig('pic/3D PDE', dpi=1200)
plt.show()


Y_origin1 = copy.deepcopy(U1) # finite difference original data
Y_origin2 = copy.deepcopy(U2)
Y_origin3 = copy.deepcopy(U3)
# gaussian noise
import random

OBS_Y1 = np.zeros([n,m])#z,t
OBS_Y2 = np.zeros([n,m])#z,t
OBS_Y3 = np.zeros([n,m])#z,t

for i in range(n):
    for j in range(m):
        if True:
            OBS_Y1[i][j] = Y_origin1[i][j] + random.gauss(0, sigma_noise)
            OBS_Y2[i][j] = Y_origin2[i][j] + random.gauss(0, sigma_noise)
            OBS_Y3[i][j] = Y_origin3[i][j] + random.gauss(0, sigma_noise)

# t, x, y, z control point
rhot = int((m-1)/(tp-1))
rhox = int((n-1)/(xp+1))

t = np.linspace(0, Max_t, tp, endpoint=True)
x = np.linspace(rhox*Max_z/(n-1), Max_z*(1-rhox/(n-1)), xp)

print('rhot', rhot, 'rhox', rhox)

nodeVector_t = [Max_t/(tp-order+1)*(i+1) for i in range(tp-order)]
nodeVector_x = [(x[0]+ (x[-1] - x[0])/(xp-order+1)*(i+1)) for i in range(xp-order)]

# grid data
sensor_grid1 = Y_origin1[rhox:(n-rhox):rhox,::rhot]
sensor_grid2 = Y_origin2[rhox:(n-rhox):rhox,::rhot]
sensor_grid3 = Y_origin3[rhox:(n-rhox):rhox,::rhot]

# print('t', t, 'x', x)
# print(sensor_grid1.shape)

# i:x j:t  flat data
sensor_flat1 = np.zeros(tp*xp)
sensor_flat2 = np.zeros(tp*xp)
sensor_flat3 = np.zeros(tp*xp)

for j in range(tp):
    for i in range(xp):
        sensor_flat1[tp * i + j] = sensor_grid1[i, j]
        sensor_flat2[tp * i + j] = sensor_grid2[i, j]
        sensor_flat3[tp * i + j] = sensor_grid3[i, j]

# B_Spline model
BS_model_1 = Bspline()
BS_model_1.fit2D(t, x, sensor_flat1, order, nodeVector_t, nodeVector_x, penalty_lambda1, penalty_lambda2, order_p)
beta_1 = BS_model_1.beta()

X_f_1 = BS_model_1.first_derivative_x(t, x, order, nodeVector_t, nodeVector_x)
T_f_1 = BS_model_1.first_derivative_t(t, x, order, nodeVector_t, nodeVector_x)
X_d_1 = BS_model_1.second_derivative_x(t, x, order, nodeVector_t, nodeVector_x)

BS_model_2 = Bspline()
BS_model_2.fit2D(t, x, sensor_flat2, order, nodeVector_t, nodeVector_x, penalty_lambda1, penalty_lambda2, order_p)
beta_2 = BS_model_2.beta()

X_f_2 = BS_model_2.first_derivative_x(t, x, order, nodeVector_t, nodeVector_x)
T_f_2 = BS_model_2.first_derivative_t(t, x, order, nodeVector_t, nodeVector_x)
X_d_2 = BS_model_2.second_derivative_x(t, x, order, nodeVector_t, nodeVector_x)

BS_model_3 = Bspline()
BS_model_3.fit2D(t, x, sensor_flat3, order, nodeVector_t, nodeVector_x, penalty_lambda1, penalty_lambda2, order_p)
beta_3 = BS_model_3.beta()

X_f_3 = BS_model_3.first_derivative_x(t, x, order, nodeVector_t, nodeVector_x)
T_f_3 = BS_model_3.first_derivative_t(t, x, order, nodeVector_t, nodeVector_x)
X_d_3 = BS_model_3.second_derivative_x(t, x, order, nodeVector_t, nodeVector_x)

H11 = BS_model_1.fitted['P1']
H12 = BS_model_1.fitted['P2']
# H13 = BS_model_1.fitted['P3']
H21 = BS_model_2.fitted['P1']
H22 = BS_model_2.fitted['P2']
# H23 = BS_model_2.fitted['H3']
H31 = BS_model_3.fitted['P1']
H32 = BS_model_3.fitted['P2']
# H33 = BS_model_3.fitted['H3']

#1st derivative of t
fst_deri1 = (T_f_1@beta_1).reshape(xp, tp)
fst_deri2 = (T_f_2@beta_2).reshape(xp, tp)
fst_deri3 = (T_f_3@beta_3).reshape(xp, tp)
# for i in range(xp):
#     plt.plot(fst_deri1[i])
#     plt.title('First derivatives of t1')
# plt.show()
# for i in range(xp):
#     plt.plot(fst_deri2[i])
#     plt.title('First derivatives of t2')
# plt.show()
# for i in range(xp):
#     plt.plot(fst_deri3[i])
#     plt.title('First derivatives of t3')
# plt.show()

# 1st derivative of x
fsz_deri1 = (X_f_1 @ beta_1).reshape(xp, tp)
fsz_deri2 = (X_f_2 @ beta_2).reshape(xp, tp)
fsz_deri3 = (X_f_3 @ beta_3).reshape(xp, tp)

# for i in range(tp):
#     plt.plot(fsz_deri1[:, i])
#     plt.title('First derivatives of x')
# plt.show()
# for i in range(tp):
#     plt.plot(fsz_deri2[:, i])
#     plt.title('First derivatives of x')
# plt.show()
# for i in range(tp):
#     plt.plot(fsz_deri3[:, i])
#     plt.title('First derivatives of x')
# plt.show()

# 2nd derivative of x
scdz_deri1 = (X_d_1 @ beta_1).reshape(xp, tp)
scdz_deri2 = (X_d_2 @ beta_2).reshape(xp, tp)
scdz_deri3 = (X_d_3 @ beta_3).reshape(xp, tp)

# for i in range(tp):
#     plt.plot(scdz_deri1[:,i])
#     plt.title('Second derivatives of x1')
# plt.show()
# for i in range(tp):
#     plt.plot(scdz_deri2[:,i])
#     plt.title('Second derivatives of x2')
# plt.show()
# for i in range(tp):
#     plt.plot(scdz_deri3[:,i])
#     plt.title('Second derivatives of x3')
# plt.show()

t11_ev0 = theta_11[::rhot]
t12_ev0 = theta_12[::rhot]
t13_ev0 = theta_13[::rhot]

con11 = t11_ev0
con12 = t12_ev0
con13 = t13_ev0
for i in range(xp-1):
    con11 = np.concatenate((con11,t11_ev0),axis=0)
    con12 = np.concatenate((con12,t12_ev0),axis=0)
    con13 = np.concatenate((con13,t13_ev0),axis=0)

t21_ev0 = theta_21[::rhot]
t22_ev0 = theta_22[::rhot]
t23_ev0 = theta_23[::rhot]

con21 = t21_ev0
con22 = t22_ev0
con23 = t23_ev0
for i in range(xp-1):
    con21 = np.concatenate((con21,t21_ev0),axis=0)
    con22 = np.concatenate((con22,t22_ev0),axis=0)
    con23 = np.concatenate((con23,t23_ev0),axis=0)

t31_ev0 = theta_31[::rhot]
t32_ev0 = theta_32[::rhot]
t33_ev0 = theta_33[::rhot]

con31 = t31_ev0
con32 = t32_ev0
con33 = t33_ev0
for i in range(xp-1):
    con31 = np.concatenate((con31,t31_ev0),axis=0)
    con32 = np.concatenate((con32,t32_ev0),axis=0)
    con33 = np.concatenate((con33,t33_ev0),axis=0)

# z1
zeta_ob1 = T_f_1@beta_1 - np.multiply(con11,X_d_1@beta_1) - np.multiply(con12, X_f_1@beta_1) - np.multiply(con13, BS_model_1.Y_hat())#BS_model_.predict_B_2D(tn, xn)@beta_2
# print(zeta_ob)
print('mean_zeta', np.mean(zeta_ob1))
print('mean_abs_zeta', np.mean(np.abs(zeta_ob1)))
print('max_zeta', np.max(zeta_ob1))
print('min_zeta', np.min(zeta_ob1))
print('rmse_zeta', np.sqrt(zeta_ob1.T@zeta_ob1/(X_f_1.shape[0])))
print('SST', zeta_ob1.T@zeta_ob1)

zeta_ob2 = T_f_2@beta_2 - np.multiply(con21, X_d_2@beta_2) - np.multiply(con22, X_f_2@beta_2) - np.multiply(con23, BS_model_2.Y_hat())#BS_model_.predict_B_2D(tn, xn)@beta_2
# print(zeta_ob)
print('mean_zeta', np.mean(zeta_ob2))
print('mean_abs_zeta', np.mean(np.abs(zeta_ob2)))
print('max_zeta', np.max(zeta_ob2))
print('min_zeta', np.min(zeta_ob2))
print('rmse_zeta', np.sqrt(zeta_ob2.T@zeta_ob2/(X_f_2.shape[0])))
print('SST', zeta_ob2.T@zeta_ob2)

zeta_ob3 = T_f_3@beta_3 - np.multiply(con31, X_d_3@beta_3) - np.multiply(con32, X_f_3@beta_3) - np.multiply(con33, BS_model_3.Y_hat())#BS_model_.predict_B_2D(tn, xn)@beta_2
# print(zeta_ob)
print('mean_zeta', np.mean(zeta_ob3))
print('mean_abs_zeta', np.mean(np.abs(zeta_ob3)))
print('max_zeta', np.max(zeta_ob3))
print('min_zeta', np.min(zeta_ob3))
print('rmse_zeta', np.sqrt(zeta_ob3.T@zeta_ob3/(X_f_3.shape[0])))
print('SST', zeta_ob3.T@zeta_ob3)

rhot2 = int((m-1)/(nt-1))
rhox2 = int((n-1-2*rhox)/(nx-1))
rhox2, rhot2

tn = np.linspace(0, Max_t, nt)
xn = np.linspace(x[0], x[-1], nx)
xn

pre_1 = BS_model_1.predict_B_2D(tn, xn)@beta_1

nodeVector_tp = [Max_t/(tp-order+1)*(i+1) for i in range(tp-order)]
nodeVector_xp = [(x[0]+ (x[-1] - x[0])/(xp-order+1)*(i+1)) for i in range(xp-order)]

X_f_1pre = BS_model_1.first_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)
T_f_1pre = BS_model_1.first_derivative_t(tn, xn, order, nodeVector_tp, nodeVector_xp)
X_d_1pre = BS_model_1.second_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)

pre_2 = BS_model_2.predict_B_2D(tn, xn)@beta_2

X_f_2pre = BS_model_2.first_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)
T_f_2pre = BS_model_2.first_derivative_t(tn, xn, order, nodeVector_tp, nodeVector_xp)
X_d_2pre = BS_model_2.second_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)

pre_3 = BS_model_3.predict_B_2D(tn, xn)@beta_3

X_f_3pre = BS_model_3.first_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)
T_f_3pre = BS_model_3.first_derivative_t(tn, xn, order, nodeVector_tp, nodeVector_xp)
X_d_3pre = BS_model_3.second_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)

validation_1 = np.empty(nt*nx)
validation_2 = np.empty(nt*nx)
validation_3 = np.empty(nt*nx)
            
for l in range(0,nt):
    for i in range(0, nx):
        validation_1[nt * i + l] = OBS_Y1[rhox+rhox2*i, rhot2*l]#+20
        validation_2[nt * i + l] = OBS_Y2[rhox+rhox2*i, rhot2*l]#+20
        validation_3[nt * i + l] = OBS_Y3[rhox+rhox2*i, rhot2*l]#+20

err1 = []
err2 = []
err3 = []
for i in range(nt*nx):
    err1.append(validation_1[i] - pre_1[i])
    err2.append(validation_2[i] - pre_2[i])
    err3.append(validation_3[i] - pre_3[i])

pre_res1 = np.empty([nx,nt])
pre_res2 = np.empty([nx,nt])
pre_res3 = np.empty([nx,nt])
for l in range(nt):
    for i in range(nx):
        pre_res1[i, l] = pre_1[nt * i + l]
        pre_res2[i, l] = pre_2[nt * i + l]
        pre_res3[i, l] = pre_3[nt * i + l]

Xv, Yv = np.meshgrid(xn, tn)
#
# figv1 = plt.figure()
# axv1 = figv1.add_subplot(projection="3d")
# axv1.set_xlabel("Z")
# axv1.set_ylabel("T")
# axv1.set_zlabel("Simulation")
# axv1.plot_surface(Xv, Yv, pre_res1.T, cmap="cool", linewidths=0.1)#
# # axv1.plot_surface(Xv, Yv, pre_res1.T-OBS_Y1[rhox:(n-rhox):rhox2,::rhot2].T, cmap="cool", linewidths=0.1)#
#
# figv2 = plt.figure()
# axv2 = figv2.add_subplot(projection="3d")
# axv2.set_xlabel("Z")
# axv2.set_ylabel("T")
# axv2.set_zlabel("Simulation")
# # axv2.plot_surface(Xv, Yv, pre_res2, cmap="cool", linewidths=0.1)#
# axv2.plot_surface(Xv, Yv, OBS_Y2[rhox:(n-rhox):rhox2,::rhot2].T, cmap="cool", linewidths=0.1)#
#
# figv3 = plt.figure()
# axv3 = figv3.add_subplot(projection="3d")
# axv3.set_xlabel("Z")
# axv3.set_ylabel("T")
# axv3.set_zlabel("Simulation")
# # axv3.plot_surface(Xv, Yv, pre_res3, cmap="cool", linewidths=0.1)#
# axv3.plot_surface(Xv, Yv, pre_res3.T-OBS_Y3[rhox:(n-rhox):rhox2,::rhot2].T, cmap="cool", linewidths=0.1)#

# plt.show()

# plt.plot((pre_res1.T-OBS_Y1.T[::rhot2,rhox:(n-rhox):rhox2]))
# plt.show()

print(np.max(pre_res1.T-OBS_Y1.T[::rhot2,rhox:(n-rhox):rhox2]))
print(np.min(pre_res1.T-OBS_Y1.T[::rhot2,rhox:(n-rhox):rhox2]))

# plt.plot((pre_res2.T-OBS_Y2.T[::rhot2,rhox:(n-rhox):rhox2]))
# plt.show()
print(np.max(pre_res2.T-OBS_Y2.T[::rhot2,rhox:(n-rhox):rhox2]))
print(np.min(pre_res2.T-OBS_Y2.T[::rhot2,rhox:(n-rhox):rhox2]))

# plt.plot((pre_res3.T-OBS_Y3.T[::rhot2,rhox:(n-rhox):rhox2]))
# plt.show()
print(np.max(pre_res3.T-OBS_Y3.T[::rhot2,rhox:(n-rhox):rhox2]))
print(np.min(pre_res3.T-OBS_Y3.T[::rhot2,rhox:(n-rhox):rhox2]))

tnn = np.linspace(0, Max_t, nt)
# tnn = np.linspace(0, 70, 71)
B_theta = theta11_bs.build_spline_mat(tnn, theta11_bs.settings['K'], theta11_bs.settings['p'], theta11_bs.settings['tau'])
print(B_theta.shape)

theta1_node1 = theta11_bs.fitted['b']
theta1_node2 = theta12_bs.fitted['b']
theta1_node3 = theta13_bs.fitted['b']
theta2_node1 = theta21_bs.fitted['b']
theta2_node2 = theta22_bs.fitted['b']
theta2_node3 = theta23_bs.fitted['b']
theta3_node1 = theta31_bs.fitted['b']
theta3_node2 = theta32_bs.fitted['b']
theta3_node3 = theta33_bs.fitted['b']


theta_node = np.array([[theta1_node1, theta1_node2, theta1_node3], [theta2_node1, theta2_node2, theta2_node3], [theta3_node1, theta3_node2, theta3_node3]])


np.save('input_simulation/B_mat_TV_1.npy',BS_model_1.predict_B_2D(tn, xn))
np.save('input_simulation/beta_TV_1.npy',beta_1)
np.save('input_simulation/Y_TV_1.npy',validation_1)

np.save('input_simulation/X_f_TV_1.npy',X_f_1pre)
np.save('input_simulation/T_f_TV_1.npy',T_f_1pre)
np.save('input_simulation/X_d_TV_1.npy',X_d_1pre)
np.save('input_simulation/B_mat_TV_2.npy',BS_model_2.predict_B_2D(tn, xn))
np.save('input_simulation/beta_TV_2.npy',beta_2)
np.save('input_simulation/Y_TV_2.npy',validation_2)

np.save('input_simulation/X_f_TV_2.npy',X_f_2pre)
np.save('input_simulation/T_f_TV_2.npy',T_f_2pre)
np.save('input_simulation/X_d_TV_2.npy',X_d_2pre)
np.save('input_simulation/B_mat_TV_3.npy',BS_model_3.predict_B_2D(tn, xn))
np.save('input_simulation/beta_TV_3.npy',beta_3)
np.save('input_simulation/Y_TV_3.npy',validation_3)

np.save('input_simulation/X_f_TV_3.npy',X_f_3pre)
np.save('input_simulation/T_f_TV_3.npy',T_f_3pre)
np.save('input_simulation/X_d_TV_3.npy',X_d_3pre)
np.save('input_simulation/B_psi.npy',B_theta)
np.save('input_simulation/psi_node.npy',theta_node)
np.save('input_simulation/H11.npy', H11)
np.save('input_simulation/H12.npy', H12)

np.save('input_simulation/H21.npy', H21)
np.save('input_simulation/H22.npy', H22)

np.save('input_simulation/H31.npy', H31)
np.save('input_simulation/H32.npy', H32)


# test data
nt = 201
nx = 161
rhot2 = int((m - 1) / (nt - 1))
rhox2 = int((n - 1 - 2 * rhox) / (nx - 1))
print('rhot2', rhot2, 'rhox2', rhox2)

tn = np.linspace(0, Max_t, nt)
xn = np.linspace(x[0], x[-1], nx)
print('xn', xn)

pre_1 = BS_model_1.predict_B_2D(tn, xn) @ beta_1
B_temp = BS_model_1.predict_B_2D(tn, xn)

nodeVector_tp = [Max_t / (tp - order + 1) * (i + 1) for i in range(tp - order)]
nodeVector_xp = [(x[0] + (x[-1] - x[0]) / (xp - order + 1) * (i + 1)) for i in range(xp - order)]

X_f_1pre = BS_model_1.first_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)
T_f_1pre = BS_model_1.first_derivative_t(tn, xn, order, nodeVector_tp, nodeVector_xp)
X_d_1pre = BS_model_1.second_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)

pre_2 = BS_model_2.predict_B_2D(tn, xn) @ beta_2

X_f_2pre = BS_model_2.first_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)
T_f_2pre = BS_model_2.first_derivative_t(tn, xn, order, nodeVector_tp, nodeVector_xp)
X_d_2pre = BS_model_2.second_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)

pre_3 = BS_model_3.predict_B_2D(tn, xn) @ beta_3

X_f_3pre = BS_model_3.first_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)
T_f_3pre = BS_model_3.first_derivative_t(tn, xn, order, nodeVector_tp, nodeVector_xp)
X_d_3pre = BS_model_3.second_derivative_x(tn, xn, order, nodeVector_tp, nodeVector_xp)

validation_1 = np.empty(nt * nx)
validation_2 = np.empty(nt * nx)
validation_3 = np.empty(nt * nx)

for l in range(0, nt):
    for i in range(0, nx):
        validation_1[nt * i + l] = OBS_Y1[rhox + rhox2 * i, rhot2 * l]  # +20
        validation_2[nt * i + l] = OBS_Y2[rhox + rhox2 * i, rhot2 * l]  # +20
        validation_3[nt * i + l] = OBS_Y3[rhox + rhox2 * i, rhot2 * l]  # +20

err1 = []
err2 = []
err3 = []
for i in range(nt * nx):
    err1.append(validation_1[i] - pre_1[i])
    err2.append(validation_2[i] - pre_2[i])
    err3.append(validation_3[i] - pre_3[i])

pre_res1 = np.empty([nx, nt])
pre_res2 = np.empty([nx, nt])
pre_res3 = np.empty([nx, nt])
for l in range(nt):
    for i in range(nx):
        pre_res1[i, l] = pre_1[nt * i + l]
        pre_res2[i, l] = pre_2[nt * i + l]
        pre_res3[i, l] = pre_3[nt * i + l]
        
Xv, Yv = np.meshgrid(xn, tn)

# figv1 = plt.figure()
# axv1 = figv1.add_subplot(projection="3d")
# axv1.set_xlabel("Z")
# axv1.set_ylabel("T")
# axv1.set_zlabel("Simulation")
# axv1.plot_surface(Xv, Yv, pre_res1.T, cmap="cool", linewidths=0.1)  #
# axv1.set_title('PDE 1')
#
# figv2 = plt.figure()
# axv2 = figv2.add_subplot(projection="3d")
# axv2.set_xlabel("Z")
# axv2.set_ylabel("T")
# axv2.set_zlabel("Simulation")
# axv2.plot_surface(Xv, Yv, OBS_Y2[rhox:(n - rhox):rhox2, ::rhot2].T, cmap="cool", linewidths=0.1)  #
#
# figv3 = plt.figure()
# axv3 = figv3.add_subplot(projection="3d")
# axv3.set_xlabel("Z")
# axv3.set_ylabel("T")
# axv3.set_zlabel("Simulation")
# axv3.plot_surface(Xv, Yv, pre_res3.T - OBS_Y3[rhox:(n - rhox):rhox2, ::rhot2].T, cmap="cool", linewidths=0.1)  #
#
# plt.show()
print(np.max(pre_res1.T - OBS_Y1.T[::rhot2, rhox:(n-rhox):rhox2]))
print(np.min(pre_res1.T - OBS_Y1.T[::rhot2, rhox:(n-rhox):rhox2]))

print(np.max(pre_res2.T - OBS_Y2.T[::rhot2, rhox:(n-rhox):rhox2]))
print(np.min(pre_res2.T - OBS_Y2.T[::rhot2, rhox:(n-rhox):rhox2]))

print(np.max(pre_res3.T - OBS_Y3.T[::rhot2, rhox:(n-rhox):rhox2]))
print(np.min(pre_res3.T - OBS_Y3.T[::rhot2, rhox:(n-rhox):rhox2]))

tnn = np.linspace(0, Max_t, nt)
B_theta = theta11_bs.build_spline_mat(tnn, theta11_bs.settings['K'], theta11_bs.settings['p'],
                                      theta11_bs.settings['tau'])
print(B_theta.shape)

theta1_node1 = theta11_bs.fitted['b']
theta1_node2 = theta12_bs.fitted['b']
theta1_node3 = theta13_bs.fitted['b']
theta2_node1 = theta21_bs.fitted['b']
theta2_node2 = theta22_bs.fitted['b']
theta2_node3 = theta23_bs.fitted['b']
theta3_node1 = theta31_bs.fitted['b']
theta3_node2 = theta32_bs.fitted['b']
theta3_node3 = theta33_bs.fitted['b']

theta_node = np.array([[theta1_node1, theta1_node2, theta1_node3], [theta2_node1, theta2_node2, theta2_node3],
                       [theta3_node1, theta3_node2, theta3_node3]])

np.save('input_pre/B_mat_TV_1.npy',BS_model_1.predict_B_2D(tn, xn))
np.save('input_pre/beta_TV_1.npy',beta_1)
np.save('input_pre/Y_TV_1.npy',validation_1)

np.save('input_pre/X_f_TV_1.npy',X_f_1pre)
np.save('input_pre/T_f_TV_1.npy',T_f_1pre)
np.save('input_pre/X_d_TV_1.npy',X_d_1pre)
np.save('input_pre/B_mat_TV_2.npy',BS_model_2.predict_B_2D(tn, xn))
np.save('input_pre/beta_TV_2.npy',beta_2)
np.save('input_pre/Y_TV_2.npy',validation_2)

np.save('input_pre/X_f_TV_2.npy',X_f_2pre)
np.save('input_pre/T_f_TV_2.npy',T_f_2pre)
np.save('input_pre/X_d_TV_2.npy',X_d_2pre)
np.save('input_pre/B_mat_TV_3.npy',BS_model_3.predict_B_2D(tn, xn))
np.save('input_pre/beta_TV_3.npy',beta_3)
np.save('input_pre/Y_TV_3.npy',validation_3)

np.save('input_pre/X_f_TV_3.npy',X_f_3pre)
np.save('input_pre/T_f_TV_3.npy',T_f_3pre)
np.save('input_pre/X_d_TV_3.npy',X_d_3pre)
np.save('input_pre/B_psi.npy',B_theta)
np.save('input_pre/psi_node.npy',theta_node)
np.save('input_pre/H11.npy', H11)
np.save('input_pre/H12.npy', H12)

np.save('input_pre/H21.npy', H21)
np.save('input_pre/H22.npy', H22)

np.save('input_pre/H31.npy', H31)
np.save('input_pre/H32.npy', H32)