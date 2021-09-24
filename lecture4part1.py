from cr3bp import EarthMoon as EM, EOMConstructor, initial_velocity
from scipy.integrate import solve_ivp, RK45
import numpy as np
import matplotlib.pyplot as plt

IC = np.array([EM.L1 + 10000/EM.l, 0, 10000/EM.l, 0, 0, 0])
IC[3:5] = initial_velocity(IC[:2], EM.L1, EM.mu)
IC = np.concatenate((IC, np.eye(6).reshape(36)))

EOMs = EOMConstructor(EM.mu, STM=True)

def event(t, y):
    return y[1] if t != 0 else -1

event.terminal = True
event.direction = -1

atol = 0.001/EM.l
traj3 = solve_ivp(EOMs, [0, 28*24*3600/EM.s], IC, method='DOP853', events=event, atol=atol, rtol=0)

c0 = IC.copy()
period = traj3.t[-1]

# algorithm
# propagate c0 until time 'period' to get cf
# from cf, we'll compute H, cfdot, STM
# if H < tol: break, otherwise
# Construct J from STM, cfdot, and our constraint
# Solve Jdq = -H (remembering to add our contraint to H)
# c0 += dq[:6]
# period += dq[-1]
# go to 1

max_iter = 50
counter = 0
tol = np.concatenate(([0.001/EM.l]*3, [0.001/EM.l*EM.s]*3))
while counter < max_iter:
    # propagate c0 until time 'period' to get cf
    c0[6:] = np.eye(6).reshape(36)
    traj = solve_ivp(EOMs, [0, period], c0, method='DOP853', atol=atol, rtol=0)
    cf = traj.y[:, -1]
    # from cf, we'll compute H, cfdot, STM
    H = cf[:6] - c0[:6]
    cfdot = EOMs(0, cf)
    STM = cf[6:].reshape((6,6))
    # if H < tol: break, otherwise
    print(counter, H)
    if all(H < tol): break
    # Construct J from STM, cfdot, and our constraint
    J = np.hstack((STM - np.eye(6), cfdot[:6].reshape((6,1))))
    J = np.vstack((J, [1, 0, 0, 0, 0, 0, 0]))
    # Solve Jdq = -H (remembering to add our contraint to H)
    H = np.concatenate((H, [0]))
    dq = np.linalg.solve(J, -H)
    c0[:6] += dq[:6]
    period += dq[-1]
    counter += 1


traj3 = solve_ivp(EOMs, [0, 31*24*3600/EM.s], c0, method='DOP853', atol=atol, rtol=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(traj3.y[0, :], traj3.y[1, :], traj3.y[2, :], 'r')
ax.plot(traj3.y[0, 0], traj3.y[1, 0], traj3.y[2, 0], 'r+')
ax.plot(*IC[:2], 'b+')

# set the axes to be equal
bound = 12000/EM.l
ax.axes.set_xlim3d(left=EM.L1 - bound, right=EM.L1 + bound)
ax.axes.set_ylim3d(bottom=-bound, top=bound)
ax.axes.set_zlim3d(bottom=-bound, top=bound)

# plot L1
ax.plot(EM.L1, 0, 0, 'k+')