from cr3bp import EarthMoon as EM, EOMConstructor, initial_velocity
from scipy.integrate import solve_ivp, DOP853
import numpy as np
import matplotlib.pyplot as plt

IC = np.array([EM.L1 + 10000/EM.l, 0, 0, 0, 0, 0])
IC[3:5] = initial_velocity(IC[:2], EM.L1, EM.mu)
IC = np.concatenate((IC, np.eye(6).reshape(36)))

EOMs = EOMConstructor(EM.mu, STM=True)

def event(t, y):
    return y[1] if t != 0 else -1

event.terminal = True
event.direction = -1

atol = 0.001/EM.l
traj3 = solve_ivp(EOMs, [0, 28*24*3600/EM.s], IC, method='DOP853', events=event, atol=atol, rtol=0)

c1 = IC.copy()
period1 = traj3.t[-1]

def velocity_targeter(c0, period):
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
    dc0dv = np.vstack((np.zeros((3,3)), np.eye(3)))
    tol = np.concatenate(([0.001/EM.l]*3, [0.001/EM.l*EM.s]*3))
    while counter < max_iter:
        # propagate c0 until time 'period' to get cf
        c0 = np.concatenate((c0[:6], np.eye(6).reshape(36)))
        traj = solve_ivp(EOMs, [0, period], c0, method='DOP853', atol=atol, rtol=0)
        cf = traj.y[:, -1]
        # from cf, we'll compute H, cfdot, STM
        H = cf[:6] - c0[:6]
        cfdot = EOMs(0, cf)
        STM = cf[6:].reshape((6,6))[:, 3:]
        # if H < tol: break, otherwise
        print(counter, H)
        if all(H < tol): break
        # Construct J from STM, cfdot, and our constraint
        J = np.hstack((STM - dc0dv, cfdot[:6].reshape((6,1))))
        # Solve Jdq = -H (remembering to add our contraint to H)
        dq = np.linalg.lstsq(J, -H)[0]
        c0[3:6] += dq[:3]
        period += dq[-1]
        counter += 1
    return c0[3:6]

c1[3:6] = velocity_targeter(c1, period1)
solver = DOP853(EOMs, 0, c1, 365*24*3600/EM.s, max_step=24*3600/EM.s, atol=atol, rtol=0)

y = [c1]
all_dvs = []
last_time_targeter_ran = 0
while solver.status == "running":
    solver.step()
    y.append(solver.y)
    if (solver.t - last_time_targeter_ran) > 24*3600/EM.s:
        print("Day", int(solver.t * EM.s/24/3600))
        last_time_targeter_ran = solver.t
        new_velocity = velocity_targeter(solver.y[:6], period1)
        dv = new_velocity - solver.y[3:6]
        dvnorm = np.linalg.norm(dv)
        if dvnorm > 0.001/EM.l*EM.s:
            print("burning")
            all_dvs.append(dvnorm)
            solver.y[3:6] = new_velocity

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

y = np.array(y).T

ax.plot(y[0, :], y[1, :], y[2, :], 'r')
ax.plot(y[0, 0], y[1, 0], y[2, 0], 'r+')
ax.plot(*IC[:2], 'b+')

# set the axes to be equal
bound = 12000/EM.l
ax.axes.set_xlim3d(left=EM.L1 - bound, right=EM.L1 + bound)
ax.axes.set_ylim3d(bottom=-bound, top=bound)
ax.axes.set_zlim3d(bottom=-bound, top=bound)

# plot L1
ax.plot(EM.L1, 0, 0, 'k+')