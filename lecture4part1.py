from cr3bp import EOMConstructor, EarthMoon as EM, initial_velocity
from scipy.integrate import solve_ivp
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
s1 = solve_ivp(eoms, [0, tf], IC, method='DOP853', events=event, atol=atol, rtol=0)

c0 = IC
T = s1.t[-1]


# algorithm:
# propagate c0 until time T to get cf
# from cf, compute H, cfdot, STM
# if norm(H) < tolerance: break, otherwise
# Construct J from STM, cfdot, and our constraint
# solve Jdq=-H (remembering to add our constraint to H)
# c0 += dq[:6]
# T += dq[-1]
# go to 1

counter = 0
max_iter = 50
tol = np.concatenate(([0.001/EM.l]*3, [0.001/EM.l*EM.s]*3))
while counter < max_iter:
    # propagate c0 until time T to get cf
    s1 = solve_ivp(eoms, [0, T], c0, method='DOP853', atol=atol, rtol=0)
    cf = s1.y[:, -1]
    # from cf, compute H, cfdot, STM
    H = cf[:6] - c0[:6]
    cfdot = eoms(0, cf)
    STM = cf[6:].reshape((6,6))
    # if norm(H) < tolerance: break, otherwise
    print(counter, np.linalg.norm(H))
    if all(H < tol): break
    # Construct J from STM, cfdot, and our constraint
    J = np.hstack((STM - np.eye(6), cfdot[:6].reshape((6,1))))
    J = np.vstack((J, [1, 0, 0, 0, 0, 0, 0]))
    # solve Jdc0=-H (remembering to add our constraint to H)
    H = np.concatenate((H, [0]))
    dq = np.linalg.solve(J, -H)
    c0[:6] += dq[:6]
    T += dq[-1]
    counter += 1

s1 = solve_ivp(eoms, [0, tf], c0, atol=atol, rtol=0)
# create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot the orbits
ax.plot(s1.y[0, :], s1.y[1, :], s1.y[2, :], 'g')
ax.plot(*IC[:3], 'b+')
ax.plot(*s1.y[:3, 0], 'g+')


# Mark the Lagrange point
ax.plot(EM.L1, 0, 0, 'k+')

# Set equal axes
bound = 50000/EM.l
ax.axes.set_xlim3d(left=EM.L1 - bound, right=EM.L1 + bound) 
ax.axes.set_ylim3d(bottom=-bound, top=bound)
ax.axes.set_zlim3d(bottom=-bound, top=bound)