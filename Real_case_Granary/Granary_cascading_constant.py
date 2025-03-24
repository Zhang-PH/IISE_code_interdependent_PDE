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


def cal_r_theta(theta_1, theta_2, theta_3, X_d, Y_d, Z_d, T_f, W):
    R_theta = theta_1 ** 2 * X_d.T @ W @ X_d + theta_2 ** 2 * Y_d.T @ W @ Y_d + theta_3 ** 2 * Z_d.T @ W @ Z_d \
              + theta_1 * theta_2 * (X_d.T @ W @ Y_d + Y_d.T @ W @ X_d) \
              + theta_1 * theta_3 * (X_d.T @ W @ Z_d + Z_d.T @ W @ X_d) \
              + theta_2 * theta_3 * (Y_d.T @ W @ Z_d + Z_d.T @ W @ Y_d) \
              - theta_1 * (T_f.T @ W @ X_d + X_d.T @ W @ T_f) \
              - theta_2 * (T_f.T @ W @ Y_d + Y_d.T @ W @ T_f) \
              - theta_3 * (T_f.T @ W @ Z_d + Z_d.T @ W @ T_f) + T_f.T @ W @ T_f
    return R_theta


def search_lambd_t(Y, B, X_d, Y_d, Z_d, T_f, W, init_theta):
    lambda_list = [1]
    G_lambda = 100000000
    best_lambda = 0
    best_beta = np.zeros(B.shape[1])
    best_theta = init_theta
    for lambd in lambda_list:
        theta = optimize_theta(Y, B, X_d, Y_d, Z_d, T_f, lambd, W, init_theta)
        beta = pre_beta(Y, B, X_d, Y_d, Z_d, T_f, theta[0], theta[1], theta[2], W, lambd)
        error = Y - B @ beta
        zeta_ob = zeta_cal(theta[0], theta[1], theta[2], T_f, X_d, Y_d, Z_d, beta)
        temp = error.T @ error + zeta_ob.T @ zeta_ob
        if temp < G_lambda:
            G_lambda = temp
            best_lambda = lambd
            best_theta = theta
            best_beta = beta

    print('best_lambda:', best_lambda)

    return best_theta, best_beta


def search_lambd_h(Y, B, X_d, Y_d, Z_d, T_f, W, init_theta):
    lambda_list = [1]
    G_lambda = 100000000
    best_lambda = 0
    best_beta = np.zeros(B.shape[1])
    best_theta = init_theta
    for lambd in lambda_list:
        theta = optimize_theta(Y, B, X_d, Y_d, Z_d, T_f, lambd, W, init_theta)
        beta = pre_beta(Y, B, X_d, Y_d, Z_d, T_f, theta[0], theta[1], theta[2], W, lambd)
        error = Y - B @ beta
        zeta_ob = zeta_cal(theta[0], theta[1], theta[2], T_f, X_d, Y_d, Z_d, beta)
        temp = error.T @ error + zeta_ob.T @ zeta_ob
        if temp < G_lambda:
            G_lambda = temp
            best_lambda = lambd
            best_theta = theta
            best_beta = beta
        
    print('best_lambda:', best_lambda)

    return best_theta, best_beta


def h_theta(theta, Y, B, X_d, Y_d, Z_d, T_f, lambd, W):
    R_theta = cal_r_theta(theta[0], theta[1], theta[2], X_d, Y_d, Z_d, T_f, W)
    temp = Y - B @ np.linalg.inv(B.T @ B + lambd * R_theta) @ B.T @ Y
    error_h = temp.T @ temp
    return error_h


def optimize_theta(Y, B, X_d, Y_d, Z_d, T_f, lambd, W, init_theta):
    res = minimize(h_theta, init_theta, args=(Y, B, X_d, Y_d, Z_d, T_f, lambd, W), method='BFGS', options={'disp': True, 'gtol': 1})
    print(res)
    return res.x


def pre_beta(Y, B, X_d, Y_d, Z_d,  T_f, theta_1, theta_2, theta_3, W, lambd):
    R_theta = cal_r_theta(theta_1, theta_2, theta_3, Z_d, X_d, Y_d, T_f, W)
    beta = np.linalg.inv(B.T @ B + lambd * R_theta) @ B.T @ Y
    return beta


def zeta_cal(tm1, tm2, tm3, T_f, X_d, Y_d, Z_d, beta):
    return (T_f - tm1*X_d - tm2*Y_d - tm3*Z_d)@beta


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

    init_theta_h = [1e-8, 1e-8, 1e-8]
    init_theta_t = [1e-8, 1e-8, 1e-8]

    xp = 3
    yp = 6
    zp = 3
    tp = 91

    w_x = gen_simpson_w(xp)
    w_y = gen_simpson_w(yp)
    w_z = gen_simpson_w(zp)
    w_t = gen_simpson_w(tp)
    W = np.kron(np.kron(np.kron(w_t, w_z), w_y), w_x)

    start_time_t = perf_counter()
    theta_t, beta_t = search_lambd_t(Y_t, B_t, X_d_t, Y_d_t, Z_d_t, T_f_t, W, init_theta_t)
    end_time_t = perf_counter()
    print('Time for temperature:', end_time_t - start_time_t)

    start_time_h = perf_counter()
    theta_h, beta_h = search_lambd_h(Y_h, B_h, X_d_h, Y_d_h, Z_d_h, T_f_h, W, init_theta_h)
    end_time_h = perf_counter()
    print('Time for humidity:', end_time_h - start_time_h)

    zeta_ob_t = zeta_cal(theta_t[0], theta_t[1], theta_t[2], T_f_t, X_d_t, Y_d_t, Z_d_t, beta_t)
    zeta_ob_h = zeta_cal(theta_h[0], theta_h[1], theta_h[2], T_f_h, X_d_h, Y_d_h, Z_d_h, beta_h)
    zeta_ob_t_ss = zeta_ob_t.T @ zeta_ob_t
    zeta_ob_h_ss = zeta_ob_h.T @ zeta_ob_h
    zeta_ob_ss = zeta_ob_t.T @ zeta_ob_t + zeta_ob_h.T @ zeta_ob_h

    print('theta_t:', theta_t)
    print('theta_h:', theta_h)

    print('PDE errors:')
    print('mean_zeta', 1/2*(np.mean(zeta_ob_t) + np.mean(zeta_ob_h)))
    print('mean_abs_zeta', 1/2*(np.mean(np.abs(zeta_ob_t)) + np.mean(np.abs(zeta_ob_h))))
    print('max_zeta', np.max([np.max(zeta_ob_t), np.max(zeta_ob_h)]))
    print('min_zeta', np.min([np.min(zeta_ob_t), np.min(zeta_ob_h)]))
    print('rmse_zeta', np.sqrt(zeta_ob_ss / zeta_ob_t.shape[0] / 2))
    print('SST_t', zeta_ob_t.T @ zeta_ob_t)
    print('SST_h', zeta_ob_h.T @ zeta_ob_h)
    print('SST', zeta_ob_ss)

    error_t = Y_t - B_t @ beta_t
    error_h = Y_h - B_h @ beta_h
    error_ss = error_t.T @ error_t + error_h.T @ error_h

    print('B_Spline Errors:')
    print('mean_BSpline', 1/2*(np.mean(error_t) + np.mean(error_h)))
    print('mean_abs_BSpline', 1/2*(np.mean(np.abs(error_t)) + np.mean(np.abs(error_h))))
    print('max_BSpline', np.max([np.max(error_t), np.max(error_h)]))
    print('min_BSpline', np.min([np.min(error_t), np.min(error_h)]))
    print('rmse_BSpline', np.sqrt(error_ss / error_t.shape[0] / 2))
    print('SST_t', error_t.T @ error_t)
    print('SST_h', error_h.T @ error_h)
    print('SST', error_ss)

    np.save('output_cascading_constant/beta_temp.npy', beta_t)
    np.save('output_cascading_constant/beta_hum.npy', beta_h)
    np.save('output_cascading_constant/theta_temp.npy', theta_t)
    np.save('output_cascading_constant/theta_hum.npy', theta_h)


