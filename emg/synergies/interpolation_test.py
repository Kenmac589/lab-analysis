import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
x = np.arange(10)
y = np.sin(x)
cs = CubicSpline(x, y)
xs = np.arange(-0.5, 9.6, 0.1)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(xs, cs(xs), label="S")
ax.set_xlim(-0.5, 9.5)
ax.legend(loc='lower left', ncol=2)
plt.show()

from scipy import interpolate

def f(x):
    x_points = [ 0, 1, 2, 3, 4, 5]
    y_points = [12,14,22,39,58,77]

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

print(f(1.25))


from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

x = [-5,-4.19,-3.54,-3.31,-2.56,-2.31,-1.66,-0.96,-0.22,0.62,1.21,3]
y = [-0.01,0.01,0.03,0.04,0.07,0.09,0.16,0.28,0.45,0.65,0.77,1]

#ascending order
if np.any(np.diff(x) < 0):
    indexes = np.argsort(x).astype(int)
    x = np.array(x)[indexes]
    y = np.array(y)[indexes]

f = CubicSpline(x, y, bc_type='natural')
x_new = np.linspace(min(x), max(x), 100)
y_new = f(x_new)

plt.plot(x_new, y_new)
plt.scatter(x, y)
plt.title('Cubic Spline Interpolation')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
rng = np.random.default_rng()
x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
spl = InterpolatedUnivariateSpline(x, y)
plt.plot(x, y, 'ro', ms=5)
xs = np.linspace(-3, 3, 1000)
plt.plot(xs, spl(xs), 'g', lw=3, alpha=0.7)
plt.show()
