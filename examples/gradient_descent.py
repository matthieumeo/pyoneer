import numpy as np
from scipy.linalg import svd, eigvals, eigvalsh
import matplotlib.pyplot as plt

A = np.array([[3, 0.], [1., 1.]])
W = np.diag([1, 20])
tau1 = .1 / (2 * np.max(svd(A, compute_uv=False)) ** 2)
tau2 = .1 / (2 * np.max(eigvalsh(np.sqrt(W) @ A.transpose() @ A @ np.sqrt(W))))
b = np.array([250., 220.])
x = np.linspace(-10, 210, 1000)
X, Y = np.meshgrid(x, x)
Z = b[None, None, :, None] - A @ np.stack((X, Y), axis=-1)[..., None]
Z = np.linalg.norm(np.squeeze(Z), axis=-1) ** 2
plt.figure(1)
plt.contour(X, Y, Z, 100)


z1 = np.zeros(shape=(2,))
z2 = np.zeros(shape=(2,))
Z1 = [z1]
Z2 = [z2]
for i in range(10000):
    z1 = z1 - 2 * tau1 * A.transpose() @ (A @ z1 - b)
    z2 = z2 - 2 * tau2 * W @ (A.transpose() @ (A @ z2 - b))
    Z1.append(z1)
    Z2.append(z2)
Z1 = np.stack(Z1, axis=-1)
Z2 = np.stack(Z2, axis=-1)

plt.figure(1)
plt.plot(Z1[0], Z1[-1], 'o-')
plt.plot(Z2[0], Z2[-1], 'o-')
plt.axis('equal')

