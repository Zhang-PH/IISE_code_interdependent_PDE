#%%
import numpy as np

# 公共数据加载
Y1 = np.load('input_pre/Y_TV_1.npy')
B1 = np.load('input_pre/B_mat_TV_1.npy')

beta1_ini = np.load('input_pre/beta_TV_1.npy')
Y1_hat = B1@beta1_ini
X_f_1 = np.load('input_pre/X_f_TV_1.npy')
X_d_1 = np.load('input_pre/X_d_TV_1.npy')
T_f_1 = np.load('input_pre/T_f_TV_1.npy')

Y2 = np.load('input_pre/Y_TV_2.npy')
B2 = np.load('input_pre/B_mat_TV_2.npy')

beta2_ini = np.load('input_pre/beta_TV_2.npy')
Y2_hat = B2@beta2_ini
X_f_2 = np.load('input_pre/X_f_TV_2.npy')
X_d_2 = np.load('input_pre/X_d_TV_2.npy')
T_f_2 = np.load('input_pre/T_f_TV_2.npy')

Y3 = np.load('input_pre/Y_TV_3.npy')
B3 = np.load('input_pre/B_mat_TV_3.npy')

beta3_ini = np.load('input_pre/beta_TV_3.npy')
Y3_hat = B3@beta3_ini
X_f_3 = np.load('input_pre/X_f_TV_3.npy')
X_d_3 = np.load('input_pre/X_d_TV_3.npy')
T_f_3 = np.load('input_pre/T_f_TV_3.npy')

B_psi = np.load('input_pre/B_psi.npy')

# IntTVP
beta1_hat = np.load('output_3in1_simulation/beta1_burn.npy')[-1]
beta2_hat = np.load('output_3in1_simulation/beta2_burn.npy')[-1]
beta3_hat = np.load('output_3in1_simulation/beta3_burn.npy')[-1]
psi_11_hat = np.load('output_3in1_simulation/psi_11_burn.npy')[-1]
psi_12_hat = np.load('output_3in1_simulation/psi_12_burn.npy')[-1]
psi_13_hat = np.load('output_3in1_simulation/psi_13_burn.npy')[-1]
psi_21_hat = np.load('output_3in1_simulation/psi_21_burn.npy')[-1]
psi_22_hat = np.load('output_3in1_simulation/psi_22_burn.npy')[-1]
psi_23_hat = np.load('output_3in1_simulation/psi_23_burn.npy')[-1]
psi_31_hat = np.load('output_3in1_simulation/psi_31_burn.npy')[-1]
psi_32_hat = np.load('output_3in1_simulation/psi_32_burn.npy')[-1]
psi_33_hat = np.load('output_3in1_simulation/psi_33_burn.npy')[-1]


def psi_mat(t_cur, B_psi):
    temp = B_psi @ t_cur
    con = np.array([temp for i in range(161)])
    return con.reshape(161 * temp.shape[0])


def zeta_cal(tm1, tm2, tm3, T_f, X_d, X_f, beta, B_psi, B_mat):
    tm1_mat = psi_mat(tm1, B_psi)
    tm2_mat = psi_mat(tm2, B_psi)
    tm3_mat = psi_mat(tm3, B_psi)
    return T_f @ beta - np.multiply(tm1_mat, X_d@beta) - np.multiply(tm2_mat, X_f@beta) - np.multiply(tm3_mat, B_mat@beta)


zeta_ob1 = zeta_cal(psi_11_hat, psi_12_hat, psi_13_hat, T_f_1, X_d_1, X_f_1, beta1_hat, B_psi, B1)
zeta_ob2 = zeta_cal(psi_21_hat, psi_22_hat, psi_23_hat, T_f_2, X_d_2, X_f_2, beta2_hat, B_psi, B1)
zeta_ob3 = zeta_cal(psi_31_hat, psi_32_hat, psi_33_hat, T_f_3, X_d_3, X_f_3, beta3_hat, B_psi, B1)
zeta_ob_ss = zeta_ob1.T @ zeta_ob1 + zeta_ob2.T @ zeta_ob2 + zeta_ob3.T @ zeta_ob3

print('PDE errors:')
print('mean_zeta', 1/3*(np.mean(zeta_ob1) + np.mean(zeta_ob2) + np.mean(zeta_ob3)))
print('mean_abs_zeta', 1/3*(np.mean(np.abs(zeta_ob1)) + np.mean(np.abs(zeta_ob2)) + np.mean(np.abs(zeta_ob3))))
print('max_zeta', np.max([np.max(zeta_ob1), np.max(zeta_ob2), np.max(zeta_ob3)]))
print('min_zeta', np.min([np.min(zeta_ob1), np.min(zeta_ob2), np.min(zeta_ob3)]))
print('rmse_zeta', np.sqrt(zeta_ob_ss/zeta_ob1.shape[0]/3))
print('SST', zeta_ob_ss)

error_bs1 = Y1-B1@beta1_hat
error_bs2 = Y2-B2@beta2_hat
error_bs3 = Y3-B3@beta3_hat
error_ss = error_bs1.T@error_bs1 + error_bs2.T@error_bs2 + error_bs3.T@error_bs3

print('B_Spline Errors:')
print('mean_BSpline', 1/3*(np.mean(error_bs1) + np.mean(error_bs2) + np.mean(error_bs3)))
print('mean_abs_BSpline', 1/3*(np.mean(np.abs(error_bs1)) + np.mean(np.abs(error_bs2)) + np.mean(np.abs(error_bs3))))
print('max_BSpline', np.max([np.max(error_bs1), np.max(error_bs2), np.max(error_bs3)]))
print('min_BSpline', np.min([np.min(error_bs1), np.min(error_bs2), np.min(error_bs3)]))
print('rmse_BSpline', np.sqrt(error_ss/error_bs1.shape[0]/3))
print('SST_BSpline', error_ss)

dt = 39
dx = 31

l_test_error_1 = np.empty(25)
l_test_error_2 = np.empty(25)
l_test_error_3 = np.empty(25)
l_test_error = np.empty(25)
l_test_zeta_1 = np.empty(25)
l_test_zeta_2 = np.empty(25)
l_test_zeta_3 = np.empty(25)
l_test_zeta = np.empty(25)
l_test_total_1 = np.empty(25)
l_test_total_2 = np.empty(25)
l_test_total_3 = np.empty(25)
l_test_total = np.empty(25)


for p in range(0, 5):
    for q in range(0, 5):
        test_zeta_1 = np.empty(dt * dx)
        test_zeta_2 = np.empty(dt * dx)
        test_zeta_3 = np.empty(dt * dx)
        test_error_1 = np.empty(dt * dx)
        test_error_2 = np.empty(dt * dx)
        test_error_3 = np.empty(dt * dx)

        for l in range(0, dt):
            for i in range(0, dx):
                test_zeta_1[dt * i + l] = zeta_ob1[201 * ((dx + 1) * p + 1 + i) + (dt + 1) * q + l + 1]
                test_zeta_2[dt * i + l] = zeta_ob2[201 * ((dx + 1) * p + 1 + i) + (dt + 1) * q + l + 1]
                test_zeta_3[dt * i + l] = zeta_ob3[201 * ((dx + 1) * p + 1 + i) + (dt + 1) * q + l + 1]
                test_error_1[dt * i + l] = error_bs1[201 * ((dx + 1) * p + 1 + i) + (dt + 1) * q + l + 1]
                test_error_2[dt * i + l] = error_bs2[201 * ((dx + 1) * p + 1 + i) + (dt + 1) * q + l + 1]
                test_error_3[dt * i + l] = error_bs3[201 * ((dx + 1) * p + 1 + i) + (dt + 1) * q + l + 1]

        test_error_1_ss = test_error_1.T @ test_error_1
        test_error_2_ss = test_error_2.T @ test_error_2
        test_error_3_ss = test_error_3.T @ test_error_3
        test_error_ss = test_error_1_ss + test_error_2_ss + test_error_3_ss
        test_zeta_1_ss = test_zeta_1.T @ test_zeta_1
        test_zeta_2_ss = test_zeta_2.T @ test_zeta_2
        test_zeta_3_ss = test_zeta_3.T @ test_zeta_3

        l_test_error_1[5 * p + q] = test_error_1_ss
        l_test_error_2[5 * p + q] = test_error_2_ss
        l_test_error_3[5 * p + q] = test_error_3_ss
        l_test_error[5 * p + q] = test_error_ss
        l_test_zeta_1[5 * p + q] = test_zeta_1_ss
        l_test_zeta_2[5 * p + q] = test_zeta_2_ss
        l_test_zeta_3[5 * p + q] = test_zeta_3_ss
        l_test_zeta[5 * p + q] = test_zeta_1_ss + test_zeta_2_ss + test_zeta_3_ss
        l_test_total_1[5 * p + q] = test_error_1_ss + test_zeta_1_ss
        l_test_total_2[5 * p + q] = test_error_2_ss + test_zeta_2_ss
        l_test_total_3[5 * p + q] = test_error_3_ss + test_zeta_3_ss
        l_test_total[5 * p + q] = test_error_ss + test_zeta_1_ss + test_zeta_2_ss + test_zeta_3_ss

print('test_error_1_mean:', np.mean(l_test_error_1))
print('test_error_1_std:', np.std(l_test_error_1))
print('test_error_2_mean:', np.mean(l_test_error_2))
print('test_error_2_std:', np.std(l_test_error_2))
print('test_error_3_mean:', np.mean(l_test_error_3))
print('test_error_3_std:', np.std(l_test_error_3))
print('test_error_mean:', np.mean(l_test_error))
print('test_error_std:', np.std(l_test_error))
print('test_zeta_1_mean:', np.mean(l_test_zeta_1))
print('test_zeta_1_std:', np.std(l_test_zeta_1))
print('test_zeta_2_mean:', np.mean(l_test_zeta_2))
print('test_zeta_2_std:', np.std(l_test_zeta_2))
print('test_zeta_3_mean:', np.mean(l_test_zeta_3))
print('test_zeta_3_std:', np.std(l_test_zeta_3))
print('test_zeta_mean:', np.mean(l_test_zeta))
print('test_zeta_std:', np.std(l_test_zeta))
print('test_total_1_mean:', np.mean(l_test_total_1))
print('test_total_1_std:', np.std(l_test_total_1))
print('test_total_2_mean:', np.mean(l_test_total_2))
print('test_total_2_std:', np.std(l_test_total_2))
print('test_total_3_mean:', np.mean(l_test_total_3))
print('test_total_3_std:', np.std(l_test_total_3))
print('test_total_mean:', np.mean(l_test_total))
print('test_total_std:', np.std(l_test_total))
