#!/usr/bin/python3
import casadi as cd
from dynamics import unicycle_ddt
from os import system


T = 5.0
N = 40

DT = T / N

def generate_collocation_functions(n_states, n_controls):
    XK = cd.MX.sym('XK', 5)
    XK1 = cd.MX.sym('XK1', 5)
    XDOTK = cd.MX.sym('XDOTK', 5)
    XDOTK1 = cd.MX.sym('XDOTK1', 5)
    UK = cd.MX.sym('UK', 2)
    UK1 = cd.MX.sym('UK1', 2)

    X_tc = 1. / 2. * (XK + XK1) + DT / 8. * (XDOTK - XDOTK1)
    f_x_tc = cd.Function('f_x_tc', [XK, XK1, XDOTK, XDOTK1], [X_tc])

    Xdot_tc = -3. / (2. * DT) * (XK - XK1) - 1. / 4. * (XDOTK + XDOTK1)
    f_xdot_tc = cd.Function('f_xdot_tc', [XK, XK1, XDOTK, XDOTK1], [Xdot_tc])

    U_tc = 1. / 2. * (UK + UK1)
    f_u_tc = cd.Function('f_u_tc', [UK, UK1], [U_tc])

    return (f_x_tc, f_xdot_tc, f_u_tc)


# defining loss function -- want to be in desired cone
#target_prediction = cd.MX.sym('target_pred', 3*40)
target_state = cd.MX.sym('target', 3)
#target_state = target_prediction[:3]
tracker_state = cd.MX.sym('tracker_state', 5)
control_input = cd.MX.sym('ctrl', 2)

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

target_heading_terminal = cd.vertcat(cd.cos(target_state[2]), cd.sin(next_target_state[2]))

cos_theta_terminal = cd.dot(x_diff_heading_terminal, target_heading_terminal)

cost_terminal = 1 * (-cos_theta_terminal + 0.01 * (cd.norm_2(x_diff_terminal)**2 - 1)**2)

l_terminal = cd.Function('l_terminal', [next_target_state, tracker_state], [cost_terminal])

##

f_x_tc, f_xdot_tc, f_u_tc = generate_collocation_functions(5, 2)

state = cd.MX.sym('state', 5)
control = cd.MX.sym('control', 2)

f = cd.Function('f', [state, control], [unicycle_ddt(state, control, None, True)])


loss = 0
w = []
w0 = []
lbw = []
ubw = []
g = []
lbg = []
ubg = []

X0 = [.1,.1,.1,.1,.1]

X0_sym = cd.MX.sym('X0', 5)
Xk = X0_sym
#w += [Xk]
#lbw += X0
#ubw += X0
#w0 += X0


Uk = cd.MX.sym('U0', 2)
w += [Uk]
lbw += [-1, -2]
ubw += [1, 2]
w0 += [.1, .1]


for k in range(N):
    loss += target_1_weights[k] * l(target_state, Xk, Uk)
    loss += target_2_weights[k] * l(next_target_state, Xk, Uk)
    
    Xk_next = cd.MX.sym('X' + str(k+1), 5)
    w += [Xk_next]
    lbw += [-10, -10, -cd.inf, 0, -2]
    ubw += [10, 10, cd.inf, 3, 2]
    w0 += [-0.5, -0.5, .1, .1, .1]


    Uk_next = cd.MX.sym('U' + str(k+1), 2)
    w += [Uk_next]
    lbw += [-1, -2]
    ubw += [1, 2]
    w0 += [.1, .1]

    Xdot_k = f(Xk, Uk)
    Xdot_k1 = f(Xk_next, Uk_next)
    x_col = f_x_tc(Xk, Xk_next, Xdot_k, Xdot_k1)
    u_col = f_u_tc(Uk, Uk_next)
    xdot_col = f_xdot_tc(Xk, Xk_next, Xdot_k, Xdot_k1)

    g += [xdot_col - f(x_col, u_col)]
    lbg += [0, 0, 0, 0, 0]
    ubg += [0, 0, 0, 0, 0]

    Xk = Xk_next
    Uk = Uk_next

loss += l_terminal(next_target_state, Xk)

prob = {'f': loss, 'x': cd.vertcat(*w), 'g': cd.vertcat(*g), 'p': cd.vertcat(X0_sym, target_state, next_target_state, target_1_weights, target_2_weights)}
solver = cd.nlpsol('solver', 'ipopt', prob)

target_position = 0

weights_1 = cd.DM.zeros(40)
weights_1[:20] = 1
weights_2 = cd.DM.zeros(40)
weights_2[20:] = 1
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(-1, -1, 0, 0, 0, 3,3,0, -1, 1, 0, weights_1, weights_2))
w_opt = sol['x']

x = []
y = []
for k in range(N):
    x.append(w_opt[7*k + 2])
    y.append(w_opt[7*k + 3])

import matplotlib.pyplot as plt
plt.plot(x,y, marker='o')
#plt.plot(x[:20],y[:20])
plt.show()

if True:
    solver.generate_dependencies('nlp.c')
    system('gcc -fPIC -shared -O2 nlp.c -o nlp.so')

solver_comp = cd.nlpsol('solver', 'ipopt', './nlp.so')
sol = solver_comp(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(X0, 3,3,cd.pi/2.0, -1, -1, 0))
print(sol['x'])


#print(w_opt)
