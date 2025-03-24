import numpy as np
from scipy.optimize import minimize
from time import perf_counter


def gen_simpson_w(n):
    temp = n // 2
    w = np.zeros(n)
    for i in range(temp):
        w[2 * i + 1] = 4
        w[2 * i] = 2
    w[0] = 1
    w[-1] = 1
    print(w)
    return np.diag(w)


def cal_theta(psi, B_psi):
    psi_temp_1 = psi[:5].reshape(5, 1)
    psi_temp_2 = psi[5:10].reshape(5, 1)
    psi_temp_3 = psi[10:15].reshape(5, 1)
    temp_1 = B_psi @ psi_temp_1
    temp_2 = B_psi @ psi_temp_2
    temp_3 = B_psi @ psi_temp_3
    con_1 = np.array([temp_1 for i in range(33)])
    con_2 = np.array([temp_2 for i in range(33)])
    con_3 = np.array([temp_3 for i in range(33)])
    theta_t_1 = np.diag(con_1.reshape(33 * temp_1.shape[0]))
    theta_t_2 = np.diag(con_2.reshape(33 * temp_2.shape[0]))
    theta_t_3 = np.diag(con_3.reshape(33 * temp_3.shape[0]))
    return theta_t_1, theta_t_2, theta_t_3


def cal_r_theta(theta_1, theta_2, theta_3, B, X_d, X_f, T_f, W):
    R_theta = X_d.T @ theta_1 @ W @ theta_1 @ X_d + X_f.T @ theta_2 @ W @ theta_2 @ X_f + B.T @ theta_3 @ W @ theta_3 @ B \
              + (X_d.T @ theta_1 @ W @ theta_2 @ X_f + X_f.T @ theta_2 @ W @ theta_1 @ X_d) \
              + (X_d.T @ theta_1 @ W @ theta_3 @ B + B.T @ theta_3 @ W @ theta_1 @ X_d) \
              + (X_f.T @ theta_2 @ W @ theta_3 @ B + B.T @ theta_3 @ W @ theta_2 @ X_f) \
              - (T_f.T @ W @ theta_1 @ X_d + X_d.T @ theta_1 @ W @ T_f) \
              - (T_f.T @ W @ theta_2 @ X_f + X_f.T @ theta_2 @ W @ T_f) \
              - (T_f.T @ W @ theta_3 @ B + B.T @ theta_3 @ W @ T_f) + T_f.T @ W @ T_f
    return R_theta


def search_lambd(B_psi, Y, B, X_d, X_f, T_f, W, init_psi):
    lambda_list = [0.05]
    G_lambda = 100000000
    best_lambda = 0
    best_beta = np.zeros(B.shape[1])
    best_psi = init_psi
    for lambd in lambda_list:
        psi = optimize_theta(B_psi, Y, B, X_d, X_f, T_f, lambd, W, init_psi)
        beta = pre_beta(Y, B, X_d, X_f, T_f, psi, W, lambd)
        error = Y - B @ beta
        zeta_ob = zeta_cal(psi, T_f, X_d, X_f, beta, B_psi, B)
        temp = error.T @ error + zeta_ob.T @ zeta_ob
        if temp < G_lambda:
            G_lambda = temp
            best_lambda = lambd
            best_psi = psi
            best_beta = beta

    print('best_lambda:', best_lambda)

    return best_psi, best_beta


def h_theta(psi, B_psi, Y, B, X_d, X_f, T_f, lambd, W):
    theta_1, theta_2, theta_3 = cal_theta(psi, B_psi)
    R_theta = cal_r_theta(theta_1, theta_2, theta_3, B, X_d, X_f, T_f, W)
    temp = Y - B @ np.linalg.inv(B.T @ B + lambd * R_theta) @ B.T @ Y
    error_h = temp.T @ temp
    return error_h


def optimize_theta(B_psi, Y, B, X_d, X_f, T_f, lambd, W, init_psi):
    bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None),(0, None), (0, None), (0, None), (0, None),(0, None), (0, None), (0, None))
    res = minimize(h_theta, init_psi, args=(B_psi, Y, B, X_d, X_f, T_f, lambd, W), method='Nelder-Mead', bounds=bnds, options={'disp': True, 'xatol': 1e-5})
    print(res)
    return res.x


def pre_beta(Y, B, X_d, X_f, T_f, psi, W, lambd):
    theta_1, theta_2, theta_3 = cal_theta(psi, B_psi)
    R_theta = cal_r_theta(theta_1, theta_2, theta_3, B, X_d, X_f, T_f, W)
    beta = np.linalg.inv(B.T @ B + lambd * R_theta) @ B.T @ Y
    return beta


def psi_mat(t_cur, B_psi):
    temp = B_psi @ t_cur
    con = np.array([temp for i in range(33)])
    return con.reshape(33 * temp.shape[0])


def zeta_cal(psi, T_f, X_d, X_f, beta, B_psi, B_mat):
    tm1 = psi[:5].reshape(5, 1)
    tm2 = psi[5:10].reshape(5, 1)
    tm3 = psi[10:15].reshape(5, 1)
    tm1_mat = psi_mat(tm1, B_psi)
    tm2_mat = psi_mat(tm2, B_psi)
    tm3_mat = psi_mat(tm3, B_psi)
    return T_f @ beta - np.multiply(tm1_mat, X_d@beta) - np.multiply(tm2_mat, X_f@beta) - np.multiply(tm3_mat, B_mat@beta)


if __name__ == '__main__':
    # true value
    Y1 = np.load('input_simulation/Y_TV_1.npy')
    B1 = np.load('input_simulation/B_mat_TV_1.npy')

    beta1_ini = np.load('input_simulation/beta_TV_1.npy')
    Y1_hat = B1 @ beta1_ini
    X_f_1 = np.load('input_simulation/X_f_TV_1.npy')
    X_d_1 = np.load('input_simulation/X_d_TV_1.npy')
    T_f_1 = np.load('input_simulation/T_f_TV_1.npy')

    Y2 = np.load('input_simulation/Y_TV_2.npy')
    B2 = np.load('input_simulation/B_mat_TV_2.npy')

    beta2_ini = np.load('input_simulation/beta_TV_2.npy')
    Y2_hat = B2 @ beta2_ini
    X_f_2 = np.load('input_simulation/X_f_TV_2.npy')
    X_d_2 = np.load('input_simulation/X_d_TV_2.npy')
    T_f_2 = np.load('input_simulation/T_f_TV_2.npy')

    Y3 = np.load('input_simulation/Y_TV_3.npy')
    B3 = np.load('input_simulation/B_mat_TV_3.npy')

    beta3_ini = np.load('input_simulation/beta_TV_3.npy')
    Y3_hat = B3 @ beta3_ini
    X_f_3 = np.load('input_simulation/X_f_TV_3.npy')
    X_d_3 = np.load('input_simulation/X_d_TV_3.npy')
    T_f_3 = np.load('input_simulation/T_f_TV_3.npy')

    B_psi = np.load('input_simulation/B_psi.npy')
    psi_node = np.load('input_simulation/psi_node.npy')

    H11 = np.load('input_simulation/H11.npy')
    H12 = np.load('input_simulation/H12.npy')
    H13 = np.load('input_simulation/H13.npy')
    H21 = np.load('input_simulation/H21.npy')
    H22 = np.load('input_simulation/H22.npy')
    H23 = np.load('input_simulation/H23.npy')
    H31 = np.load('input_simulation/H31.npy')
    H32 = np.load('input_simulation/H32.npy')
    H33 = np.load('input_simulation/H33.npy')

    init_psi_1 = np.zeros(15)
    init_psi_2 = np.zeros(15)
    init_psi_3 = np.zeros(15)

    xp = 33
    tp = 41
    w_x = gen_simpson_w(xp)
    w_t = gen_simpson_w(tp)
    W = np.kron(w_t, w_x)
    iterations = 0

    start_time1 = perf_counter()
    psi_1, beta1 = search_lambd(B_psi, Y1, B1, X_d_1, X_f_1, T_f_1, W, init_psi_1)
    end_time1 = perf_counter()
    print("PDE1 burn_in stage time:", end_time1 - start_time1, "s")
    start_time2 = perf_counter()
    psi_2, beta2 = search_lambd(B_psi, Y2, B2, X_d_2, X_f_2, T_f_2, W, init_psi_2)
    end_time2 = perf_counter()
    print("PDE2 burn_in stage time:", end_time2 - start_time2, "s")
    start_time3 = perf_counter()
    psi_3, beta3 = search_lambd(B_psi, Y3, B3, X_d_3, X_f_3, T_f_3, W, init_psi_3)
    end_time3 = perf_counter()
    print("PDE3 burn_in stage time:", end_time3 - start_time3, "s")

    zeta_ob1 = zeta_cal(psi_1, T_f_1, X_d_1, X_f_1, beta1, B_psi, B1)
    zeta_ob2 = zeta_cal(psi_2, T_f_2, X_d_2, X_f_2, beta2, B_psi, B2)
    zeta_ob3 = zeta_cal(psi_3, T_f_3, X_d_3, X_f_3, beta3, B_psi, B3)
    zeta_ob1_ss = zeta_ob1.T @ zeta_ob1
    zeta_ob2_ss = zeta_ob2.T @ zeta_ob2
    zeta_ob3_ss = zeta_ob3.T @ zeta_ob3
    zeta_ob_ss = zeta_ob1.T @ zeta_ob1 + zeta_ob2.T @ zeta_ob2 + zeta_ob3.T @ zeta_ob3

    print('PDE errors:')
    print('mean_zeta', 1 / 3 * (np.mean(zeta_ob1) + np.mean(zeta_ob2) + np.mean(zeta_ob3)))
    print('mean_abs_zeta', 1 / 3 * (np.mean(np.abs(zeta_ob1)) + np.mean(np.abs(zeta_ob2)) + np.mean(np.abs(zeta_ob3))))
    print('max_zeta', np.max([np.max(zeta_ob1), np.max(zeta_ob2), np.max(zeta_ob3)]))
    print('min_zeta', np.min([np.min(zeta_ob1), np.min(zeta_ob2), np.min(zeta_ob3)]))
    print('rmse_zeta', np.sqrt(zeta_ob_ss / zeta_ob1.shape[0] / 3))
    print('SST_1', zeta_ob1_ss)
    print('SST_2', zeta_ob2_ss)
    print('SST_3', zeta_ob3_ss)
    print('SST', zeta_ob_ss)

    error_bs1 = Y1 - B1 @ beta1
    error_bs2 = Y2 - B2 @ beta2
    error_bs3 = Y3 - B3 @ beta3
    error_bs1_ss = error_bs1.T @ error_bs1
    error_bs2_ss = error_bs2.T @ error_bs2
    error_bs3_ss = error_bs3.T @ error_bs3
    error_ss = error_bs1.T @ error_bs1 + error_bs2.T @ error_bs2 + error_bs3.T @ error_bs3

    print('B_Spline Errors:')
    print('mean_BSpline', 1 / 3 * (np.mean(error_bs1) + np.mean(error_bs2) + np.mean(error_bs3)))
    print('mean_abs_BSpline',
          1 / 3 * (np.mean(np.abs(error_bs1)) + np.mean(np.abs(error_bs2)) + np.mean(np.abs(error_bs3))))
    print('max_BSpline', np.max([np.max(error_bs1), np.max(error_bs2), np.max(error_bs3)]))
    print('min_BSpline', np.min([np.min(error_bs1), np.min(error_bs2), np.min(error_bs3)]))
    print('rmse_BSpline', np.sqrt(error_ss / error_bs1.shape[0] / 3))
    print('SST_BSpline_1', error_bs1_ss)
    print('SST_BSpline_2', error_bs2_ss)
    print('SST_BSpline_3', error_bs3_ss)
    print('SST_BSpline', error_ss)

    np.save('output_cascading/beta1.npy', beta1)
    np.save('output_cascading/beta2.npy', beta2)
    np.save('output_cascading/beta3.npy', beta3)
    np.save('output_cascading/psi_11.npy', psi_1[:5].reshape(5, 1))
    np.save('output_cascading/psi_12.npy', psi_1[5:10].reshape(5, 1))
    np.save('output_cascading/psi_13.npy', psi_1[10:15].reshape(5, 1))
    np.save('output_cascading/psi_21.npy', psi_2[:5].reshape(5, 1))
    np.save('output_cascading/psi_22.npy', psi_2[5:10].reshape(5, 1))
    np.save('output_cascading/psi_23.npy', psi_2[10:15].reshape(5, 1))
    np.save('output_cascading/psi_31.npy', psi_3[:5].reshape(5, 1))
    np.save('output_cascading/psi_32.npy', psi_3[5:10].reshape(5, 1))
    np.save('output_cascading/psi_33.npy', psi_3[10:15].reshape(5, 1))
