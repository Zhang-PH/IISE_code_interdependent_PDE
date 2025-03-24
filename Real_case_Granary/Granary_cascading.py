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
    con_1 = np.array([temp_1 for i in range(54)])
    con_2 = np.array([temp_2 for i in range(54)])
    con_3 = np.array([temp_3 for i in range(54)])
    theta_t_1 = np.diag(con_1.reshape(54 * temp_1.shape[0]))
    theta_t_2 = np.diag(con_2.reshape(54 * temp_2.shape[0]))
    theta_t_3 = np.diag(con_3.reshape(54 * temp_3.shape[0]))
    return theta_t_1, theta_t_2, theta_t_3


def cal_r_theta(theta_1, theta_2, theta_3, X_d, Y_d, Z_d, T_f, W):
    R_theta = X_d.T @ theta_1 @ W @ theta_1 @ X_d + Y_d.T @ theta_2 @ W @ theta_2 @ Y_d + Z_d.T @ theta_3 @ W @ theta_3 @ Z_d \
              + (X_d.T @ theta_1 @ W @ theta_2 @ Y_d + Y_d.T @ theta_2 @ W @ theta_1 @ X_d) \
              + (X_d.T @ theta_1 @ W @ theta_3 @ Z_d + Z_d.T @ theta_3 @ W @ theta_1 @ X_d) \
              + (Y_d.T @ theta_2 @ W @ theta_3 @ Z_d + Z_d.T @ theta_3 @ W @ theta_2 @ Y_d) \
              - (T_f.T @ W @ theta_1 @ X_d + X_d.T @ theta_1 @ W @ T_f) \
              - (T_f.T @ W @ theta_2 @ Y_d + Y_d.T @ theta_2 @ W @ T_f) \
              - (T_f.T @ W @ theta_3 @ Z_d + Z_d.T @ theta_3 @ W @ T_f) + T_f.T @ W @ T_f
    return R_theta


def search_lambd_t(B_psi, Y, B, X_d, Y_d, Z_d, T_f, W, init_psi):
    lambda_list = [1]
    G_lambda = 100000000
    best_lambda = 0
    best_beta = np.zeros(B.shape[1])
    best_psi = init_psi
    for lambd in lambda_list:
        psi = optimize_theta(B_psi, Y, B, X_d, Y_d, Z_d, T_f, lambd, W, init_psi)
        beta = pre_beta(Y, B, X_d, Y_d, Z_d, T_f, psi, W, lambd)
        error = Y - B @ beta
        zeta_ob = zeta_cal(psi, T_f, X_d, Y_d, Z_d, beta, B_psi)
        temp = error.T @ error + zeta_ob.T @ zeta_ob
        if temp < G_lambda:
            G_lambda = temp
            best_lambda = lambd
            best_psi = psi
            best_beta = beta

    print('best_lambda:', best_lambda)

    return best_psi, best_beta


def search_lambd_h(B_psi, Y, B, X_d, Y_d, Z_d, T_f, W, init_psi):
    lambda_list = [1]
    G_lambda = 100000000
    best_lambda = 0
    best_beta = np.zeros(B.shape[1])
    best_psi = init_psi
    for lambd in lambda_list:
        psi = optimize_theta(B_psi, Y, B, X_d, Y_d, Z_d, T_f, lambd, W, init_psi)
        beta = pre_beta(Y, B, X_d, Y_d, Z_d, T_f, psi, W, lambd)
        error = Y - B @ beta
        zeta_ob = zeta_cal(psi, T_f, X_d, Y_d, Z_d, beta, B_psi)
        temp = error.T @ error + zeta_ob.T @ zeta_ob
        if temp < G_lambda:
            G_lambda = temp
            best_lambda = lambd
            best_psi = psi
            best_beta = beta

    print('best_lambda:', best_lambda)

    return best_psi, best_beta


def h_theta(psi, B_psi, Y, B, X_d, Y_d, Z_d, T_f, lambd, W):
    theta_1, theta_2, theta_3 = cal_theta(psi, B_psi)
    R_theta = cal_r_theta(theta_1, theta_2, theta_3, X_d, Y_d, Z_d, T_f, W)
    temp = Y - B @ np.linalg.inv(B.T @ B + lambd * R_theta) @ B.T @ Y
    error_h = temp.T @ temp
    return error_h


def optimize_theta(B_psi, Y, B, X_d, Y_d, Z_d, T_f, lambd, W, init_psi):
    res = minimize(h_theta, init_psi, args=(B_psi, Y, B, X_d, Y_d, Z_d, T_f, lambd, W), method='BFGS', 
                   options={'disp': True, 'gtol': 1})
    print(res)
    return res.x


def pre_beta(Y, B, X_d, Y_d, Z_d, T_f, psi, W, lambd):
    theta_1, theta_2, theta_3 = cal_theta(psi, B_psi)
    R_theta = cal_r_theta(theta_1, theta_2, theta_3, X_d, Y_d, Z_d, T_f, W)
    beta = np.linalg.inv(B.T @ B + lambd * R_theta) @ B.T @ Y
    return beta


def psi_mat(t_cur, B_psi):
    temp = B_psi @ t_cur
    con = np.array([temp for i in range(54)])
    return con.reshape(54 * temp.shape[0])


def zeta_cal(psi, T_f, X_d, Y_d, Z_d, beta, B_psi):
    tm1 = psi[:5].reshape(5, 1)
    tm2 = psi[5:10].reshape(5, 1)
    tm3 = psi[10:15].reshape(5, 1)
    tm1_mat = psi_mat(tm1, B_psi)
    tm2_mat = psi_mat(tm2, B_psi)
    tm3_mat = psi_mat(tm3, B_psi)
    return T_f @ beta - np.multiply(tm1_mat, X_d @ beta) - np.multiply(tm2_mat, Y_d @ beta) - np.multiply(tm3_mat,
                                                                                                          Z_d @ beta)


if __name__ == '__main__':
    # True value
    B_psi = np.load('input/B_psi_91.npy')

    Y_t = np.load('input/Y_temp.npy')
    B_t = np.load('input/B_mat_temp.npy')

    beta_ini_t = np.load('input/beta_temp.npy')
    Y_hat_t = B_t @ beta_ini_t

    X_d_t = np.load('input/X_d_temp.npy')
    Y_d_t = np.load('input/Y_d_temp.npy')
    Z_d_t = np.load('input/Z_d_temp.npy')
    T_f_t = np.load('input/T_f_temp.npy')

    Y_h = np.load('input/Y_hum.npy')
    B_h = np.load('input/B_mat_hum.npy')

    beta_ini_h = np.load('input/beta_hum.npy')
    Y_hat_h = B_h @ beta_ini_h

    X_d_h = np.load('input/X_d_hum.npy')
    Y_d_h = np.load('input/Y_d_hum.npy')
    Z_d_h = np.load('input/Z_d_hum.npy')
    T_f_h = np.load('input/T_f_hum.npy')

    init_psi_t = np.zeros(15)
    init_psi_h = np.zeros(15)

    xp = 3
    yp = 6
    zp = 3
    tp = 91

    w_x = gen_simpson_w(xp)
    w_y = gen_simpson_w(yp)
    w_z = gen_simpson_w(zp)
    w_t = gen_simpson_w(tp)
    W = np.kron(np.kron(np.kron(w_t, w_z), w_y), w_x)

    start_time1 = perf_counter()
    psi_t, beta_t = search_lambd_t(B_psi, Y_t, B_t, X_d_t, Y_d_t, Z_d_t, T_f_t, W, init_psi_t)
    end_time1 = perf_counter()
    print('Time for temperature:', end_time1 - start_time1)
    start_time2 = perf_counter()
    psi_h, beta_h = search_lambd_h(B_psi, Y_h, B_h, X_d_h, Y_d_h, Z_d_h, T_f_h, W, init_psi_h)
    end_time2 = perf_counter()
    print('Time for humidity:', end_time2 - start_time2)

    zeta_ob_t = zeta_cal(psi_t, T_f_t, X_d_t, Y_d_t, Z_d_t, beta_t, B_psi)
    zeta_ob_h = zeta_cal(psi_h, T_f_h, X_d_h, Y_d_h, Z_d_h, beta_h, B_psi)
    zata_ob_t_ss = zeta_ob_t.T @ zeta_ob_t
    zata_ob_h_ss = zeta_ob_h.T @ zeta_ob_h
    zeta_ob_ss = zata_ob_t_ss + zata_ob_h_ss

    print('PDE errors:')
    print('mean_zeta', 1 / 2 * (np.mean(zeta_ob_t) + np.mean(zeta_ob_h)))
    print('mean_abs_zeta', 1 / 2 * (np.mean(np.abs(zeta_ob_t)) + np.mean(np.abs(zeta_ob_h))))
    print('max_zeta', np.max([np.max(zeta_ob_t), np.max(zeta_ob_h)]))
    print('min_zeta', np.min([np.min(zeta_ob_t), np.min(zeta_ob_h)]))
    print('rmse_zeta', np.sqrt(zeta_ob_ss / zeta_ob_t.shape[0] / 2))
    print('SST_t', zata_ob_t_ss)
    print('SST_h', zata_ob_h_ss)
    print('SST', zeta_ob_ss)

    error_bs_t = Y_t - B_t @ beta_t
    error_bs_h = Y_h - B_h @ beta_h
    error_bs_t_ss = error_bs_t.T @ error_bs_t
    error_bs_h_ss = error_bs_h.T @ error_bs_h
    error_ss = error_bs_t_ss + error_bs_h_ss

    print('B_Spline Errors:')
    print('mean_BSpline', 1 / 2 * (np.mean(error_bs_t) + np.mean(error_bs_h)))
    print('mean_abs_BSpline', 1 / 2 * (np.mean(np.abs(error_bs_t)) + np.mean(np.abs(error_bs_h))))
    print('max_BSpline', np.max([np.max(error_bs_t), np.max(error_bs_h)]))
    print('min_BSpline', np.min([np.min(error_bs_t), np.min(error_bs_h)]))
    print('rmse_BSpline', np.sqrt(error_ss / error_bs_t.shape[0] / 2))
    print('SST_t', error_bs_t_ss)
    print('SST_h', error_bs_h_ss)
    print('SST', error_ss)

    np.save('output_cascading/beta_temp.npy', beta_t)
    np.save('output_cascading/beta_hum.npy', beta_h)
    np.save('output_cascading/psi_temp_1.npy', psi_t[:5].reshape(5, 1))
    np.save('output_cascading/psi_temp_2.npy', psi_t[5:10].reshape(5, 1))
    np.save('output_cascading/psi_temp_3.npy', psi_t[10:15].reshape(5, 1))
    np.save('output_cascading/psi_hum_1.npy', psi_h[:5].reshape(5, 1))
    np.save('output_cascading/psi_hum_2.npy', psi_h[5:10].reshape(5, 1))
    np.save('output_cascading/psi_hum_3.npy', psi_h[10:15].reshape(5, 1))

