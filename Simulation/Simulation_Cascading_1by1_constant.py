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


def cal_r_theta(theta_1, theta_2, theta_3, B, X_d, X_f, T_f, W):
    R_theta = theta_1 ** 2 * X_d.T @ W @ X_d + theta_2 ** 2 * X_f.T @ W @ X_f + theta_3 ** 2 * B.T @ W @ B \
              + theta_1 * theta_2 * (X_d.T @ W @ X_f + X_f.T @ W @ X_d) \
              + theta_1 * theta_3 * (X_d.T @ W @ B + B.T @ W @ X_d) \
              + theta_2 * theta_3 * (X_f.T @ W @ B + B.T @ W @ X_f) \
              - theta_1 * (T_f.T @ W @ X_d + X_d.T @ W @ T_f) \
              - theta_2 * (T_f.T @ W @ X_f + X_f.T @ W @ T_f) \
              - theta_3 * (T_f.T @ W @ B + B.T @ W @ T_f) + T_f.T @ W @ T_f
    return R_theta


def search_lambd(Y, B, X_d, X_f, T_f, W, init_theta):
    lambda_list = [0.0567]
    G_lambda = 100000000
    best_lambda = 0
    best_beta = np.zeros(B.shape[1])
    best_theta = init_theta
    for lambd in lambda_list:
        theta = optimize_theta(Y, B, X_d, X_f, T_f, lambd, W, init_theta)
        beta = pre_beta(Y, B, X_d, X_f, T_f, theta[0], theta[1], theta[2], W, lambd)
        error = Y - B @ beta
        zeta_ob = zeta_cal(theta[0], theta[1], theta[2], T_f, X_d, X_f, beta, B)
        temp = error.T @ error + zeta_ob.T @ zeta_ob
        if temp < G_lambda:
            G_lambda = temp
            best_lambda = lambd
            best_theta = theta
            best_beta = beta

    print('best_lambda:', best_lambda)

    return best_theta, best_beta


def h_theta(theta, Y, B, X_d, X_f, T_f, lambd, W):
    R_theta = cal_r_theta(theta[0], theta[1], theta[2], B, X_d, X_f, T_f, W)
    temp = Y - B @ np.linalg.inv(B.T @ B + lambd * R_theta) @ B.T @ Y
    error_h = temp.T @ temp
    return error_h


def optimize_theta(Y, B, X_d, X_f, T_f, lambd, W, init_theta):
    res = minimize(h_theta, init_theta, args=(Y, B, X_d, X_f, T_f, lambd, W), method='BFGS', options={'disp': True, 'gtol': 1})
    print(res)
    return res.x


def pre_beta(Y, B, X_d, X_f, T_f, theta_1, theta_2, theta_3, W, lambd):
    R_theta = cal_r_theta(theta_1, theta_2, theta_3, B, X_d, X_f, T_f, W)
    beta = np.linalg.inv(B.T @ B + lambd * R_theta) @ B.T @ Y
    return beta


def zeta_cal(tm1, tm2, tm3, T_f, X_d, X_f, beta, B_mat):
    return (T_f - tm1*X_d - tm2*X_f - tm3*B_mat)@beta


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

    init_theta_1 = [0, 0, 0]
    init_theta_2 = [0, 0, 0]
    init_theta_3 = [0, 0, 0]

    xp = 33
    tp = 41
    w_x = gen_simpson_w(xp)
    w_t = gen_simpson_w(tp)
    W = np.kron(w_t, w_x)
    iterations = 0

    start_time1 = perf_counter()
    theta_1, beta1 = search_lambd(Y1, B1, X_d_1, X_f_1, T_f_1, W, init_theta_1)
    end_time1 = perf_counter()
    print("PDE1 burn_in stage time:", end_time1 - start_time1, "s")
    start_time2 = perf_counter()
    theta_2, beta2 = search_lambd(Y2, B2, X_d_2, X_f_2, T_f_2, W, init_theta_2)
    end_time2 = perf_counter()
    print("PDE2 burn_in stage time:", end_time2 - start_time2, "s")
    start_time3 = perf_counter()
    theta_3, beta3 = search_lambd(Y3, B3, X_d_3, X_f_3, T_f_3, W, init_theta_3)
    end_time3 = perf_counter()
    print("PDE3 burn_in stage time:", end_time3 - start_time3, "s")

    zeta_ob1 = zeta_cal(theta_1[0], theta_1[1], theta_1[2], T_f_1, X_d_1, X_f_1, beta1, B1)
    zeta_ob2 = zeta_cal(theta_2[0], theta_2[1], theta_2[2], T_f_2, X_d_2, X_f_2, beta2, B2)
    zeta_ob3 = zeta_cal(theta_3[0], theta_3[1], theta_3[2], T_f_3, X_d_3, X_f_3, beta3, B3)
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

    np.save('output_cascading_constant/beta1.npy', beta1)
    np.save('output_cascading_constant/beta2.npy', beta2)
    np.save('output_cascading_constant/beta3.npy', beta3)
    np.save('output_cascading_constant/theta_11.npy', theta_1[0])
    np.save('output_cascading_constant/theta_12.npy', theta_1[1])
    np.save('output_cascading_constant/theta_13.npy', theta_1[2])
    np.save('output_cascading_constant/theta_21.npy', theta_2[0])
    np.save('output_cascading_constant/theta_22.npy', theta_2[1])
    np.save('output_cascading_constant/theta_23.npy', theta_2[2])
    np.save('output_cascading_constant/theta_31.npy', theta_3[0])
    np.save('output_cascading_constant/theta_32.npy', theta_3[1])
    np.save('output_cascading_constant/theta_33.npy', theta_3[2])