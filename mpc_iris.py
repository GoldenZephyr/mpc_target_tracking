#!/usr/bin/python3
import casadi as cd
#from dynamics import unicycle_ddt
from dynamics import mpcc_ddt
from os import system


T = 5.0
N = 40

DT = T / N
npoly = 4
n_state = 6
n_control = 3

def poly_derivative(coefs):
    coef_d = cd.SX.sym('c_deriv', npoly - 1)
    # casadi's polynomial coefficient ordering is high order to low order
    for ix in range(coef_d.shape[0]):
        coef_d[ix] = (npoly - 1 - ix) * coefs[ix]
    return coef_d


def generate_contour_cost_function():
    coefs_x = cd.SX.sym('cx', npoly)
    coefs_y = cd.SX.sym('cy', npoly)
    tracker_position = cd.SX.sym('x', 2)
    contour_dist = cd.SX.sym('d')
    contour_dist_des = cd.SX.sym('d_des')
    deriv_coefs_x = poly_derivative(coefs_x)
    deriv_coefs_y = poly_derivative(coefs_y)

    w_c = cd.SX.sym('w_c')
    w_l = cd.SX.sym('w_l')
    w_d = cd.SX.sym('w_d')

    path_poly_eval = lambda cx, cy, d: cd.vertcat(cd.polyval(cx, d), cd.polyval(cy, d))
    s_theta = path_poly_eval(coefs_x, coefs_y, contour_dist)
    r_pqs = s_theta - tracker_position
    s_prime = path_poly_eval(deriv_coefs_x, deriv_coefs_y, contour_dist)
    n = s_prime / cd.norm_2(s_prime)
    res_proj = cd.mtimes([cd.transpose(r_pqs), n])
    e_l = cd.norm_2(res_proj) ** 2
    e_c = cd.norm_2(r_pqs - cd.mtimes([res_proj, n])) ** 2

    #err = cd.norm_2(r_pqs)**2
    #p = path_poly_eval(coefs_x, coefs_y, 0)
    #err = cd.norm_2(tracker_position - p)**2 
    #err = 10*cd.norm_2(tracker_position - p)**2 + 10*cd.norm_2(tracker_position - path_poly_eval(coefs_x, coefs_y, contour_dist))**2
    #err = cd.norm_2(tracker_position - path_poly_eval(coefs_x, coefs_y, contour_dist))**2
    #output = [err]

    inputs = [tracker_position, contour_dist, contour_dist_des, coefs_x, coefs_y, w_c, w_l, w_d]
    labels = ['tracker_position', 'contour_dist', 'contour_dist_des', 'cx', 'cy', 'w_c', 'w_l', 'w_d']
    output = [w_c * e_c + w_l * e_l + w_d * (contour_dist - contour_dist_des)**2]
    #control_cost = cd.Function('state_costs', inputs, output, labels, ['cost'])
    control_cost = cd.Function('state_costs', inputs, output)
    return control_cost


contour_cost_func = generate_contour_cost_function()


def generate_collocation_functions(nx, nu):
    XK = cd.MX.sym('XK', nx)
    XK1 = cd.MX.sym('XK1', nx)
    XDOTK = cd.MX.sym('XDOTK', nx)
    XDOTK1 = cd.MX.sym('XDOTK1', nx)
    UK = cd.MX.sym('UK', nu)
    UK1 = cd.MX.sym('UK1', nu)

    X_tc = 1. / 2. * (XK + XK1) + DT / 8. * (XDOTK - XDOTK1)
    f_x_tc = cd.Function('f_x_tc', [XK, XK1, XDOTK, XDOTK1], [X_tc])

    Xdot_tc = -3. / (2. * DT) * (XK - XK1) - 1. / 4. * (XDOTK + XDOTK1)
    f_xdot_tc = cd.Function('f_xdot_tc', [XK, XK1, XDOTK, XDOTK1], [Xdot_tc])

    U_tc = 1. / 2. * (UK + UK1)
    f_u_tc = cd.Function('f_u_tc', [UK, UK1], [U_tc])

    return (f_x_tc, f_xdot_tc, f_u_tc)


# defining loss function -- want to be in desired cone
target_prediction = cd.MX.sym('target_pred', 3*40)
target_state = cd.MX.sym('target', 3)
#target_state = target_prediction[:3]
tracker_state = cd.MX.sym('tracker_state', n_state)
control_input = cd.MX.sym('ctrl', n_control)

spline_coef_x = cd.MX.sym('spline_x', npoly)
spline_coef_y = cd.MX.sym('spline_y', npoly)

target_1_weights = cd.MX.sym('target_1_weights', 40)
target_2_weights = cd.MX.sym('target_2_weights', 40)

x_diff = tracker_state[:2] - target_state[:2]
x_diff_heading = x_diff / cd.norm_2(x_diff)

target_heading = cd.vertcat(cd.cos(target_state[2]), cd.sin(target_state[2]))

cos_theta = cd.dot(x_diff_heading, target_heading)
cost = -cos_theta + cd.sqrt(3)/2.0 + .001 * cd.norm_2(control_input)**2 + .01 * (cd.norm_2(x_diff)**2 - 1)**2
#cost = cd.norm_2(x_diff)**2 + .1 * cd.norm_2(control_input)**2
#cost = cd.norm_2(tracker_state[:2] - target_state[:2])**2 + .01*cd.norm_2(control_input)**2


l = cd.Function('l', [target_state, tracker_state, control_input], [cost])

# terminal cost
next_target_state = cd.MX.sym('target', 3)
x_diff_terminal = tracker_state[:2] - next_target_state[:2]
x_diff_heading_terminal = x_diff_terminal / cd.norm_2(x_diff_terminal)

target_heading_terminal = cd.vertcat(cd.cos(next_target_state[2]), cd.sin(next_target_state[2]))

cos_theta_terminal = cd.dot(x_diff_heading_terminal, target_heading_terminal)

cost_terminal = 1 * (-cos_theta_terminal + 0.01 * (cd.norm_2(x_diff_terminal)**2 - 1)**2)

l_terminal = cd.Function('l_terminal', [next_target_state, tracker_state], [cost_terminal])

##

f_x_tc, f_xdot_tc, f_u_tc = generate_collocation_functions(n_state, n_control)

state = cd.MX.sym('state', n_state)
control = cd.MX.sym('control', n_control)

f = cd.Function('f', [state, control], [mpcc_ddt(state, control, None, True)])


loss = 0
w = []
w0 = []
lbw = []
ubw = []
g = []
lbg = []
ubg = []

X0 = [.1,.1,.1,.1,.1, 0]

X0_sym = cd.MX.sym('X0', n_state)
Xk = X0_sym


Uk = cd.MX.sym('U0', n_control)
w += [Uk]
lbw += [-1, -2, -.1]
ubw += [1, 2, .1]
w0 += [.1, .1, .1]


for k in range(N):
    #loss += target_1_weights[k] * l(target_prediction[3*k:3*k+3], Xk, Uk)
    #loss += target_2_weights[k] * l(next_target_state, Xk, Uk)

    #tracker_position, contour_dist, contour_dist_des, coefs_x, coefs_y, w_c, w_l, w_d
    loss += contour_cost_func(Xk[:2], Xk[5], 1, spline_coef_x, spline_coef_y, 1, 1, 1)
    loss += .01*cd.norm_2(Uk)**2


    Xk_next = cd.MX.sym('X' + str(k+1), n_state)
    w += [Xk_next]
    lbw += [-10, -10, -cd.inf, 0, -2, 0]
    ubw += [10, 10, cd.inf, 3, 2, 1]
    w0 += [-0.5, -0.5, .1, .1, .1, .1]


    Uk_next = cd.MX.sym('U' + str(k+1), n_control)
    w += [Uk_next]
    lbw += [-1, -2, -.01]
    ubw += [1, 2, .01]
    w0 += [.1, .1, .1]

    Xdot_k = f(Xk, Uk)
    Xdot_k1 = f(Xk_next, Uk_next)
    x_col = f_x_tc(Xk, Xk_next, Xdot_k, Xdot_k1)
    u_col = f_u_tc(Uk, Uk_next)
    xdot_col = f_xdot_tc(Xk, Xk_next, Xdot_k, Xdot_k1)

    g += [xdot_col - f(x_col, u_col)]
    lbg += [0, 0, 0, 0, 0, 0]
    ubg += [0, 0, 0, 0, 0, 0]

    Xk = Xk_next
    Uk = Uk_next

#loss += l_terminal(next_target_state, Xk)

prob = {'f': loss, 'x': cd.vertcat(*w), 'g': cd.vertcat(*g), 'p': cd.vertcat(X0_sym, target_prediction, next_target_state, target_1_weights, target_2_weights, spline_coef_x, spline_coef_y)}
solver = cd.nlpsol('solver', 'ipopt', prob)

target_position = 0

weights_1 = cd.DM.zeros(40)
weights_1[:20] = 1
weights_2 = cd.DM.zeros(40)
weights_2[20:] = 1

#spline_coef_x = cd.vertcat(0, 0, 1, 0)
#spline_coef_y = cd.vertcat(0, 1, 0, 0)

spline_coef_x = cd.DM.ones(npoly)
spline_coef_y = cd.DM.ones(npoly)

target_prediction = cd.DM.zeros(3*40)
for k in range(40):
    target_prediction[3*k] = 3
    target_prediction[3*k + 1] = 3
    target_prediction[3*k + 2] = 0

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(.1, .1, 0, 0, 0, 0, target_prediction, -1, 1, 0, weights_1, weights_2, spline_coef_x, spline_coef_y))
w_opt = sol['x']

x = []
y = []
for k in range(N):
    x.append(w_opt[9*k + 3])
    y.append(w_opt[9*k + 4])

import matplotlib.pyplot as plt
plt.plot(x,y, marker='o')
#plt.plot(x[:20],y[:20])
plt.show()

if True:
    solver.generate_dependencies('nlp_iris.c')
    system('gcc -fPIC -shared -O2 nlp_iris.c -o nlp_iris.so')

solver_comp = cd.nlpsol('solver', 'ipopt', './nlp_iris.so')
sol = solver_comp(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(X0, 3,3,cd.pi/2.0, -1, -1, 0))
print(sol['x'])


#print(w_opt)
