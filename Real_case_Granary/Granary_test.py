import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, gamma, norm, multivariate_normal, pearsonr

# Load the data
Y_t = np.load('input/Y_temp.npy')
B_t = np.load('input/B_mat_temp.npy')
T_f_t = np.load('input/T_f_temp.npy')
X_d_t = np.load('input/X_d_temp.npy')
Y_d_t = np.load('input/Y_d_temp.npy')
Z_d_t = np.load('input/Z_d_temp.npy')

Y_h = np.load('input/Y_hum.npy')
B_h = np.load('input/B_mat_hum.npy')
T_f_h = np.load('input/T_f_hum.npy')
X_d_h = np.load('input/X_d_hum.npy')
Y_d_h = np.load('input/Y_d_hum.npy')
Z_d_h = np.load('input/Z_d_hum.npy')

B_psi = np.load('input/B_psi_91.npy')

# int-tvp
beta_temp_1 = np.load('output_2in1/beta1_steady.npy')[-1]
beta_hum_1 = np.load('output_2in1/beta2_steady.npy')[-1]
psi_11_1 = np.load('output_2in1/psi_11_steady.npy')[-1]
psi_12_1 = np.load('output_2in1/psi_12_steady.npy')[-1]
psi_13_1 = np.load('output_2in1/psi_13_steady.npy')[-1]
psi_21_1 = np.load('output_2in1/psi_21_steady.npy')[-1]
psi_22_1 = np.load('output_2in1/psi_22_steady.npy')[-1]
psi_23_1 = np.load('output_2in1/psi_23_steady.npy')[-1]

# ind-tvp
beta_temp_2 = np.load('output_temp/beta_steady.npy')[-1]
psi_11_2 = np.load('output_temp/psi_1_steady.npy')[-1]
psi_12_2 = np.load('output_temp/psi_2_steady.npy')[-1]
psi_13_2 = np.load('output_temp/psi_3_steady.npy')[-1]
beta_hum_2 = np.load('output_hum/beta_steady.npy')[-1]
psi_21_2 = np.load('output_hum/psi_1_steady.npy')[-1]
psi_22_2 = np.load('output_hum/psi_2_steady.npy')[-1]
psi_23_2 = np.load('output_hum/psi_3_steady.npy')[-1]

# ind-cp
beta_temp_3 = np.load('output_temp_constant/beta_steady.npy')[-1]
psi_11_3 = np.load('output_temp_constant/psi_1_steady.npy')[-1]
psi_12_3 = np.load('output_temp_constant/psi_2_steady.npy')[-1]
psi_13_3 = np.load('output_temp_constant/psi_3_steady.npy')[-1]
beta_hum_3 = np.load('output_hum_constant/beta_steady.npy')[-1]
psi_21_3 = np.load('output_hum_constant/psi_1_steady.npy')[-1]
psi_22_3 = np.load('output_hum_constant/psi_2_steady.npy')[-1]
psi_23_3 = np.load('output_hum_constant/psi_3_steady.npy')[-1]

# tvp-pcm
beta_temp_4 = np.load('output_cascading/beta_temp.npy')
beta_hum_4 = np.load('output_cascading/beta_hum.npy')
psi_11_4 = np.load('output_cascading/psi_temp_1.npy')
psi_12_4 = np.load('output_cascading/psi_temp_2.npy')
psi_13_4 = np.load('output_cascading/psi_temp_3.npy')
psi_21_4 = np.load('output_cascading/psi_hum_1.npy')
psi_22_4 = np.load('output_cascading/psi_hum_2.npy')
psi_23_4 = np.load('output_cascading/psi_hum_3.npy')

# cp-pcm
beta_temp_5 = np.load('output_cascading_constant_2/beta_temp.npy')
beta_hum_5 = np.load('output_cascading_constant_2/beta_hum.npy')
theta_t = np.load('output_cascading_constant_2/theta_temp.npy')
theta_h = np.load('output_cascading_constant_2/theta_hum.npy')


def psi_mat(t_cur, B_psi):
    temp = B_psi @ t_cur
    con = np.array([temp for i in range(54)])
    return con.reshape(54 * temp.shape[0])


def zeta_cal_cp(tm1, tm2, tm3, T_f, X_d, Y_d, Z_d, beta):
    zeta = (T_f - tm1*X_d - tm2*Y_d - tm3*Z_d)@beta
    return zeta


def zeta_cal_tvp(tm1, tm2, tm3, T_f, X_d, Y_d, Z_d, beta, B_psi):
    tm1_mat = psi_mat(tm1, B_psi)
    tm2_mat = psi_mat(tm2, B_psi)
    tm3_mat = psi_mat(tm3, B_psi)
    return T_f @ beta - np.multiply(tm1_mat, X_d@beta) - np.multiply(tm2_mat, Y_d@beta) - np.multiply(tm3_mat, Z_d@beta)


# int-tvp
zeta_t_1 = zeta_cal_tvp(psi_11_1, psi_12_1, psi_13_1, T_f_t, X_d_t, Y_d_t, Z_d_t, beta_temp_1, B_psi)
zeta_h_1 = zeta_cal_tvp(psi_21_1, psi_22_1, psi_23_1, T_f_h, X_d_h, Y_d_h, Z_d_h, beta_hum_1, B_psi)
error_t_1 = Y_t - B_t @ beta_temp_1
error_h_1 = Y_h - B_h @ beta_hum_1

# ind-tvp
zeta_t_2 = zeta_cal_tvp(psi_11_2, psi_12_2, psi_13_2, T_f_t, X_d_t, Y_d_t, Z_d_t, beta_temp_2, B_psi)
zeta_h_2 = zeta_cal_tvp(psi_21_2, psi_22_2, psi_23_2, T_f_h, X_d_h, Y_d_h, Z_d_h, beta_hum_2, B_psi)
error_t_2 = Y_t - B_t @ beta_temp_2
error_h_2 = Y_h - B_h @ beta_hum_2

# ind-cp
zeta_t_3 = zeta_cal_cp(psi_11_3, psi_12_3, psi_13_3, T_f_t, X_d_t, Y_d_t, Z_d_t, beta_temp_3)
zeta_h_3 = zeta_cal_cp(psi_21_3, psi_22_3, psi_23_3, T_f_h, X_d_h, Y_d_h, Z_d_h, beta_hum_3)
error_t_3 = Y_t - B_t @ beta_temp_3
error_h_3 = Y_h - B_h @ beta_hum_3

# tvp-pcm
zeta_t_4 = zeta_cal_tvp(psi_11_4, psi_12_4, psi_13_4, T_f_t, X_d_t, Y_d_t, Z_d_t, beta_temp_4, B_psi)
zeta_h_4 = zeta_cal_tvp(psi_21_4, psi_22_4, psi_23_4, T_f_h, X_d_h, Y_d_h, Z_d_h, beta_hum_4, B_psi)
error_t_4 = Y_t - B_t @ beta_temp_4
error_h_4 = Y_h - B_h @ beta_hum_4

# cp-pcm
zeta_t_5 = zeta_cal_cp(theta_t[0], theta_t[1], theta_t[2], T_f_t, X_d_t, Y_d_t, Z_d_t, beta_temp_5)
zeta_h_5 = zeta_cal_cp(theta_h[0], theta_h[1], theta_h[2], T_f_h, X_d_h, Y_d_h, Z_d_h, beta_hum_5)
error_t_5 = Y_t - B_t @ beta_temp_5
error_h_5 = Y_h - B_h @ beta_hum_5


# set test-set
n_set = 90
n_t = 7
n_space = 54
t_total = 721
n_point = n_t * n_space

l_test_error_t_1 = np.empty(n_set)
l_test_error_t_2 = np.empty(n_set)
l_test_error_t_3 = np.empty(n_set)
l_test_error_t_4 = np.empty(n_set)
l_test_error_t_5 = np.empty(n_set)
l_test_error_h_1 = np.empty(n_set)
l_test_error_h_2 = np.empty(n_set)
l_test_error_h_3 = np.empty(n_set)
l_test_error_h_4 = np.empty(n_set)
l_test_error_h_5 = np.empty(n_set)
l_test_error_1 = np.empty(n_set)
l_test_error_2 = np.empty(n_set)
l_test_error_3 = np.empty(n_set)
l_test_error_4 = np.empty(n_set)
l_test_error_5 = np.empty(n_set)

l_test_zeta_t_1 = np.empty(n_set)
l_test_zeta_t_2 = np.empty(n_set)
l_test_zeta_t_3 = np.empty(n_set)
l_test_zeta_t_4 = np.empty(n_set)
l_test_zeta_t_5 = np.empty(n_set)
l_test_zeta_h_1 = np.empty(n_set)
l_test_zeta_h_2 = np.empty(n_set)
l_test_zeta_h_3 = np.empty(n_set)
l_test_zeta_h_4 = np.empty(n_set)
l_test_zeta_h_5 = np.empty(n_set)
l_test_zeta_1 = np.empty(n_set)
l_test_zeta_2 = np.empty(n_set)
l_test_zeta_3 = np.empty(n_set)
l_test_zeta_4 = np.empty(n_set)
l_test_zeta_5 = np.empty(n_set)

l_test_total_t_1 = np.empty(n_set)
l_test_total_t_2 = np.empty(n_set)
l_test_total_t_3 = np.empty(n_set)
l_test_total_t_4 = np.empty(n_set)
l_test_total_t_5 = np.empty(n_set)
l_test_total_h_1 = np.empty(n_set)
l_test_total_h_2 = np.empty(n_set)
l_test_total_h_3 = np.empty(n_set)
l_test_total_h_4 = np.empty(n_set)
l_test_total_h_5 = np.empty(n_set)
l_test_total_1 = np.empty(n_set)
l_test_total_2 = np.empty(n_set)
l_test_total_3 = np.empty(n_set)
l_test_total_4 = np.empty(n_set)
l_test_total_5 = np.empty(n_set)

for p in range(n_set):
    test_error_t_1 = np.empty(n_point)
    test_error_t_2 = np.empty(n_point)
    test_error_t_3 = np.empty(n_point)
    test_error_t_4 = np.empty(n_point)
    test_error_t_5 = np.empty(n_point)
    test_error_h_1 = np.empty(n_point)
    test_error_h_2 = np.empty(n_point)
    test_error_h_3 = np.empty(n_point)
    test_error_h_4 = np.empty(n_point)
    test_error_h_5 = np.empty(n_point)
    test_zeta_t_1 = np.empty(n_point)
    test_zeta_t_2 = np.empty(n_point)
    test_zeta_t_3 = np.empty(n_point)
    test_zeta_t_4 = np.empty(n_point)
    test_zeta_t_5 = np.empty(n_point)
    test_zeta_h_1 = np.empty(n_point)
    test_zeta_h_2 = np.empty(n_point)
    test_zeta_h_3 = np.empty(n_point)
    test_zeta_h_4 = np.empty(n_point)
    test_zeta_h_5 = np.empty(n_point)
    test_error_1 = np.empty(n_point)
    test_error_2 = np.empty(n_point)
    test_error_3 = np.empty(n_point)
    test_error_4 = np.empty(n_point)
    test_error_5 = np.empty(n_point)
    test_zeta_1 = np.empty(n_point)
    test_zeta_2 = np.empty(n_point)
    test_zeta_3 = np.empty(n_point)
    test_zeta_4 = np.empty(n_point)
    test_zeta_5 = np.empty(n_point)

    for l in range(0, n_t):
        for i in range(0, 54):
            test_error_t_1[n_t * i + l] = error_t_1[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_t_2[n_t * i + l] = error_t_2[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_t_3[n_t * i + l] = error_t_3[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_t_4[n_t * i + l] = error_t_4[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_t_5[n_t * i + l] = error_t_5[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_h_1[n_t * i + l] = error_h_1[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_h_2[n_t * i + l] = error_h_2[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_h_3[n_t * i + l] = error_h_3[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_h_4[n_t * i + l] = error_h_4[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_h_5[n_t * i + l] = error_h_5[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_t_1[n_t * i + l] = zeta_t_1[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_t_2[n_t * i + l] = zeta_t_2[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_t_3[n_t * i + l] = zeta_t_3[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_t_4[n_t * i + l] = zeta_t_4[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_t_5[n_t * i + l] = zeta_t_5[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_h_1[n_t * i + l] = zeta_h_1[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_h_2[n_t * i + l] = zeta_h_2[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_h_3[n_t * i + l] = zeta_h_3[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_h_4[n_t * i + l] = zeta_h_4[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_h_5[n_t * i + l] = zeta_h_5[t_total * i + (n_t + 1) * p + l + 1] ** 2

            test_error_1[n_t * i + l] = error_t_1[t_total * i + (n_t + 1) * p + l + 1] ** 2 + error_h_1[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_2[n_t * i + l] = error_t_2[t_total * i + (n_t + 1) * p + l + 1] ** 2 + error_h_2[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_3[n_t * i + l] = error_t_3[t_total * i + (n_t + 1) * p + l + 1] ** 2 + error_h_3[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_4[n_t * i + l] = error_t_4[t_total * i + (n_t + 1) * p + l + 1] ** 2 + error_h_4[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_error_5[n_t * i + l] = error_t_5[t_total * i + (n_t + 1) * p + l + 1] ** 2 + error_h_5[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_1[n_t * i + l] = zeta_t_1[t_total * i + (n_t + 1) * p + l + 1] ** 2 + zeta_h_1[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_2[n_t * i + l] = zeta_t_2[t_total * i + (n_t + 1) * p + l + 1] ** 2 + zeta_h_2[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_3[n_t * i + l] = zeta_t_3[t_total * i + (n_t + 1) * p + l + 1] ** 2 + zeta_h_3[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_4[n_t * i + l] = zeta_t_4[t_total * i + (n_t + 1) * p + l + 1] ** 2 + zeta_h_4[t_total * i + (n_t + 1) * p + l + 1] ** 2
            test_zeta_5[n_t * i + l] = zeta_t_5[t_total * i + (n_t + 1) * p + l + 1] ** 2 + zeta_h_5[t_total * i + (n_t + 1) * p + l + 1] ** 2


    l_test_error_t_1[p] = np.sum(test_error_t_1)
    l_test_error_t_2[p] = np.sum(test_error_t_2)
    l_test_error_t_3[p] = np.sum(test_error_t_3)
    l_test_error_t_4[p] = np.sum(test_error_t_4)
    l_test_error_t_5[p] = np.sum(test_error_t_5)
    l_test_error_h_1[p] = np.sum(test_error_h_1)
    l_test_error_h_2[p] = np.sum(test_error_h_2)
    l_test_error_h_3[p] = np.sum(test_error_h_3)
    l_test_error_h_4[p] = np.sum(test_error_h_4)
    l_test_error_h_5[p] = np.sum(test_error_h_5)
    l_test_zeta_t_1[p] = np.sum(test_zeta_t_1)
    l_test_zeta_t_2[p] = np.sum(test_zeta_t_2)
    l_test_zeta_t_3[p] = np.sum(test_zeta_t_3)
    l_test_zeta_t_4[p] = np.sum(test_zeta_t_4)
    l_test_zeta_t_5[p] = np.sum(test_zeta_t_5)
    l_test_zeta_h_1[p] = np.sum(test_zeta_h_1)
    l_test_zeta_h_2[p] = np.sum(test_zeta_h_2)
    l_test_zeta_h_3[p] = np.sum(test_zeta_h_3)
    l_test_zeta_h_4[p] = np.sum(test_zeta_h_4)
    l_test_zeta_h_5[p] = np.sum(test_zeta_h_5)
    l_test_total_t_1[p] = l_test_error_t_1[p] + l_test_zeta_t_1[p]
    l_test_total_t_2[p] = l_test_error_t_2[p] + l_test_zeta_t_2[p]
    l_test_total_t_3[p] = l_test_error_t_3[p] + l_test_zeta_t_3[p]
    l_test_total_t_4[p] = l_test_error_t_4[p] + l_test_zeta_t_4[p]
    l_test_total_t_5[p] = l_test_error_t_5[p] + l_test_zeta_t_5[p]
    l_test_total_h_1[p] = l_test_error_h_1[p] + l_test_zeta_h_1[p]
    l_test_total_h_2[p] = l_test_error_h_2[p] + l_test_zeta_h_2[p]
    l_test_total_h_3[p] = l_test_error_h_3[p] + l_test_zeta_h_3[p]
    l_test_total_h_4[p] = l_test_error_h_4[p] + l_test_zeta_h_4[p]
    l_test_total_h_5[p] = l_test_error_h_5[p] + l_test_zeta_h_5[p]

    l_test_error_1[p] = np.sum(test_error_1)
    l_test_error_2[p] = np.sum(test_error_2)
    l_test_error_3[p] = np.sum(test_error_3)
    l_test_error_4[p] = np.sum(test_error_4)
    l_test_error_5[p] = np.sum(test_error_5)
    l_test_zeta_1[p] = np.sum(test_zeta_1)
    l_test_zeta_2[p] = np.sum(test_zeta_2)
    l_test_zeta_3[p] = np.sum(test_zeta_3)
    l_test_zeta_4[p] = np.sum(test_zeta_4)
    l_test_zeta_5[p] = np.sum(test_zeta_5)
    l_test_total_1[p] = l_test_error_1[p] + l_test_zeta_1[p]
    l_test_total_2[p] = l_test_error_2[p] + l_test_zeta_2[p]
    l_test_total_3[p] = l_test_error_3[p] + l_test_zeta_3[p]
    l_test_total_4[p] = l_test_error_4[p] + l_test_zeta_4[p]
    l_test_total_5[p] = l_test_error_5[p] + l_test_zeta_5[p]

# random select 10 areas
n = 10
num_1 = list(np.random.choice(range(90), n, replace=False))

test_final_error_t_1 = np.empty(n)
test_final_error_t_2 = np.empty(n)
test_final_error_t_3 = np.empty(n)
test_final_error_t_4 = np.empty(n)
test_final_error_t_5 = np.empty(n)
test_final_error_h_1 = np.empty(n)
test_final_error_h_2 = np.empty(n)
test_final_error_h_3 = np.empty(n)
test_final_error_h_4 = np.empty(n)
test_final_error_h_5 = np.empty(n)
test_final_zeta_t_1 = np.empty(n)
test_final_zeta_t_2 = np.empty(n)
test_final_zeta_t_3 = np.empty(n)
test_final_zeta_t_4 = np.empty(n)
test_final_zeta_t_5 = np.empty(n)
test_final_zeta_h_1 = np.empty(n)
test_final_zeta_h_2 = np.empty(n)
test_final_zeta_h_3 = np.empty(n)
test_final_zeta_h_4 = np.empty(n)
test_final_zeta_h_5 = np.empty(n)
test_final_total_t_1 = np.empty(n)
test_final_total_t_2 = np.empty(n)
test_final_total_t_3 = np.empty(n)
test_final_total_t_4 = np.empty(n)
test_final_total_t_5 = np.empty(n)
test_final_total_h_1 = np.empty(n)
test_final_total_h_2 = np.empty(n)
test_final_total_h_3 = np.empty(n)
test_final_total_h_4 = np.empty(n)
test_final_total_h_5 = np.empty(n)


test_final_error_1 = np.empty(n)
test_final_error_2 = np.empty(n)
test_final_error_3 = np.empty(n)
test_final_error_4 = np.empty(n)
test_final_error_5 = np.empty(n)
test_final_zeta_1 = np.empty(n)
test_final_zeta_2 = np.empty(n)
test_final_zeta_3 = np.empty(n)
test_final_zeta_4 = np.empty(n)
test_final_zeta_5 = np.empty(n)
test_final_total_1 = np.empty(n)
test_final_total_2 = np.empty(n)
test_final_total_3 = np.empty(n)
test_final_total_4 = np.empty(n)
test_final_total_5 = np.empty(n)

flag = 0
for p in num_1:
    test_final_error_t_1[flag] = l_test_error_t_1[p]
    test_final_error_t_2[flag] = l_test_error_t_2[p]
    test_final_error_t_3[flag] = l_test_error_t_3[p]
    test_final_error_t_4[flag] = l_test_error_t_4[p]
    test_final_error_t_5[flag] = l_test_error_t_5[p]
    test_final_error_h_1[flag] = l_test_error_h_1[p]
    test_final_error_h_2[flag] = l_test_error_h_2[p]
    test_final_error_h_3[flag] = l_test_error_h_3[p]
    test_final_error_h_4[flag] = l_test_error_h_4[p]
    test_final_error_h_5[flag] = l_test_error_h_5[p]
    test_final_zeta_t_1[flag] = l_test_zeta_t_1[p]
    test_final_zeta_t_2[flag] = l_test_zeta_t_2[p]
    test_final_zeta_t_3[flag] = l_test_zeta_t_3[p]
    test_final_zeta_t_4[flag] = l_test_zeta_t_4[p]
    test_final_zeta_t_5[flag] = l_test_zeta_t_5[p]
    test_final_zeta_h_1[flag] = l_test_zeta_h_1[p]
    test_final_zeta_h_2[flag] = l_test_zeta_h_2[p]
    test_final_zeta_h_3[flag] = l_test_zeta_h_3[p]
    test_final_zeta_h_4[flag] = l_test_zeta_h_4[p]
    test_final_zeta_h_5[flag] = l_test_zeta_h_5[p]
    test_final_total_t_1[flag] = l_test_total_t_1[p]
    test_final_total_t_2[flag] = l_test_total_t_2[p]
    test_final_total_t_3[flag] = l_test_total_t_3[p]
    test_final_total_t_4[flag] = l_test_total_t_4[p]
    test_final_total_t_5[flag] = l_test_total_t_5[p]
    test_final_total_h_1[flag] = l_test_total_h_1[p]
    test_final_total_h_2[flag] = l_test_total_h_2[p]
    test_final_total_h_3[flag] = l_test_total_h_3[p]
    test_final_total_h_4[flag] = l_test_total_h_4[p]
    test_final_total_h_5[flag] = l_test_total_h_5[p]
    test_final_error_1[flag] = l_test_error_1[p]
    test_final_error_2[flag] = l_test_error_2[p]
    test_final_error_3[flag] = l_test_error_3[p]
    test_final_error_4[flag] = l_test_error_4[p]
    test_final_error_5[flag] = l_test_error_5[p]
    test_final_zeta_1[flag] = l_test_zeta_1[p]
    test_final_zeta_2[flag] = l_test_zeta_2[p]
    test_final_zeta_3[flag] = l_test_zeta_3[p]
    test_final_zeta_4[flag] = l_test_zeta_4[p]
    test_final_zeta_5[flag] = l_test_zeta_5[p]
    test_final_total_1[flag] = l_test_total_1[p]
    test_final_total_2[flag] = l_test_total_2[p]
    test_final_total_3[flag] = l_test_total_3[p]
    test_final_total_4[flag] = l_test_total_4[p]
    test_final_total_5[flag] = l_test_total_5[p]
    flag += 1

# output
print(np.mean(test_final_error_t_1))
print(np.mean(test_final_error_h_1))
print(np.mean(test_final_error_1))
print(np.mean(test_final_error_t_2))
print(np.mean(test_final_error_h_2))
print(np.mean(test_final_error_2))
print(np.mean(test_final_error_t_3))
print(np.mean(test_final_error_h_3))
print(np.mean(test_final_error_3))
print(np.mean(test_final_error_t_4))
print(np.mean(test_final_error_h_4))
print(np.mean(test_final_error_4))
print(np.mean(test_final_error_t_5))
print(np.mean(test_final_error_h_5))
print(np.mean(test_final_error_5))
print(np.mean(test_final_zeta_t_1))
print(np.mean(test_final_zeta_h_1))
print(np.mean(test_final_zeta_1))
print(np.mean(test_final_zeta_t_2))
print(np.mean(test_final_zeta_h_2))
print(np.mean(test_final_zeta_2))
print(np.mean(test_final_zeta_t_3))
print(np.mean(test_final_zeta_h_3))
print(np.mean(test_final_zeta_3))
print(np.mean(test_final_zeta_t_4))
print(np.mean(test_final_zeta_h_4))
print(np.mean(test_final_zeta_4))
print(np.mean(test_final_zeta_t_5))
print(np.mean(test_final_zeta_h_5))
print(np.mean(test_final_zeta_5))
print(np.mean(test_final_total_t_1))
print(np.mean(test_final_total_h_1))
print(np.mean(test_final_total_1))
print(np.mean(test_final_total_t_2))
print(np.mean(test_final_total_h_2))
print(np.mean(test_final_total_2))
print(np.mean(test_final_total_t_3))
print(np.mean(test_final_total_h_3))
print(np.mean(test_final_total_3))
print(np.mean(test_final_total_t_4))
print(np.mean(test_final_total_h_4))
print(np.mean(test_final_total_4))
print(np.mean(test_final_total_t_5))
print(np.mean(test_final_total_h_5))
print(np.mean(test_final_total_5))


