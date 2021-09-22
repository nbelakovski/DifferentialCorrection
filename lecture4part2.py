from cr3bp import EOMConstructor, EarthMoon as EM, initial_velocity
from scipy.integrate import solve_ivp, DOP853
import numpy as np
import matplotlib.pyplot as plt

# Create the initial conditions
IC = np.array([10000/EM.l + EM.L1, 0, 0, 0, 0, 0])
IC[3:5] = initial_velocity(IC[:2], EM.L1, EM.mu)

eoms = EOMConstructor(EM.mu, STM=True)
IC = np.concatenate((IC, np.eye(6).reshape(36)))

tf = 3600*24*38 / EM.seconds

def event(t, c):
    return -1 if t == 0 else c[1]

event.direction = -1
event.terminal = True

# numerically integrate the EOMs
atol=0.001/EM.l
s1 = solve_ivp(eoms, [0, tf], IC, events=event, atol=atol, rtol=0)

c1 = IC.copy()
P = s1.t[-1]

def velocity_targetter(c0, T):
    # algorithm:
    # propagate c0 until time T to get cf
    # from cf, compute H, cfdot, STM
    # if norm(H) < tolerance: break, otherwise
    # Construct J from STM, cfdot
    # solve Jdq=-H
    # c0[3:6] += dq[:3]
    # T += dq[-1]
    # go to 1
    
    counter = 0
    max_iter = 50
    tol = 0.001/EM.l
    dc0dv = np.vstack((np.zeros((3,3)), np.eye(3)))
    c0 = np.concatenate((c0[:6], np.eye(6).reshape(36)))
    while counter < max_iter:
        s1 = solve_ivp(eoms, [0, T], c0, method='DOP853', atol=atol, rtol=0)
        cf = s1.y[:, -1]
        # from cf, compute H, cfdot, STM
        H = cf[:6] - c0[:6]
        cfdot = eoms(0, cf)
        STM = cf[6:].reshape((6,6))[:, 3:]
        # if norm(H) < tolerance: break, otherwise
        if np.linalg.norm(H) < tol: break
        # Construct J from STM, cfdot
        J = np.hstack((STM - dc0dv, cfdot[:6].reshape((6,1))))
        # solve Jdc0=-H
        dq = np.linalg.lstsq(J, -H)[0]
        c0[3:6] += dq[:3]
        T += dq[-1]
        counter += 1
        print(counter, np.linalg.norm(H))
    return c0, T

## Stationkeeping
c1, P = velocity_targetter(c1, P)
solver = DOP853(eoms, 0, c1, 365*24*3600/EM.s, max_step=1*24*3600/EM.s, atol=atol, rtol=0)

y = [c1]
all_dvs = []
last_time_targetter_run = 0
while solver.status == "running":
    # print(solver.t * EM.s / 3600 / 24)
    solver.step()
    y.append(solver.y)
    if (solver.t - last_time_targetter_run) > 1*24*3600/EM.s:
        print("Day: ", int(solver.t * EM.s / 24 / 3600))
        last_time_targetter_run = solver.t
        new_velocity, P = velocity_targetter(solver.y, P)
        dv = new_velocity[3:6] - solver.y[3:6]
        dvnorm = np.linalg.norm(dv)
        if dvnorm > 0.001/EM.l*EM.s:
            print("BURNING")
            all_dvs.append(dvnorm)
            solver.y[3:6] = new_velocity[3:6]
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

y = np.array(y).T

ax.plot(y[0, :], y[1, :], y[2, :], 'b')
ax.plot(y[0, 0], y[1, 0], y[2, 0], 'b+')

# set the axes to be equal
bound = 50000/EM.l
ax.axes.set_xlim3d(left=EM.L1 - bound, right=EM.L1 + bound)
ax.axes.set_ylim3d(bottom=-bound, top=bound)
ax.axes.set_zlim3d(bottom=-bound, top=bound)

# plot L1
ax.plot(EM.L1, 0, 0, 'k+')