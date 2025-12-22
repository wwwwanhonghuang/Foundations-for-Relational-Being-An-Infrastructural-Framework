import numpy as np
from qutip import *

# Number of particles / modes
N = 3

# Local Hilbert space dimension per mode
d = 4

# Annihilation operators a_i
a_ops = []

for i in range(N):
    ops = []
    for j in range(N):
        if i == j:
            ops.append(destroy(d))
        else:
            ops.append(qeye(d))
    a_ops.append(tensor(ops))

adag_ops = [a.dag() for a in a_ops]

theta0 = {
    "omega": np.array([1.0, 1.2, 0.8])
}

Theta = np.array([
    [0.0, 0.10, 0.05],
    [0.10, 0.0, 0.08],
    [0.05, 0.08, 0.0]
])


H0 = 0

for i in range(N):
    H0 += theta0["omega"][i] * adag_ops[i] * a_ops[i]
    
Hint = 0

for i in range(N):
    for j in range(i + 1, N):
        gij = Theta[i, j]
        Hint += gij * (adag_ops[i] * a_ops[j] + adag_ops[j] * a_ops[i])
        
drive = 0
eps = [0.2, 0.0, 0.0]  # drive only mode 0

for i in range(N):
    drive += eps[i] * (a_ops[i] + adag_ops[i])
    
H = H0 + Hint + drive

gamma = 0.05
c_ops = []

for i in range(N):
    c_ops.append(np.sqrt(gamma) * a_ops[i])
    
psi0 = tensor([basis(d, 0) for _ in range(N)])
rho0 = ket2dm(psi0)

tlist = np.linspace(0, 20, 400)

e_ops = [adag_ops[i] * a_ops[i] for i in range(N)]

result = mesolve(
    H,
    rho0,
    tlist,
    c_ops,
    e_ops
)

import matplotlib.pyplot as plt

for i in range(N):
    plt.plot(tlist, result.expect[i], label=f"Mode {i}")

plt.xlabel("Time")
plt.ylabel("Occupation ⟨n⟩")
plt.legend()
plt.title("Open-system dynamics with NxN coupling matrix Θ")
plt.show()
