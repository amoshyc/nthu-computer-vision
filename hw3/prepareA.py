import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 50
xs = np.random.rand(N)
ys = 5 * xs + 4
noise = np.random.rand(N) * 0.05 - 0.025
ys = ys + noise

df = pd.DataFrame(columns=['x', 'y'])
df['x'] = xs
df['y'] = ys
df.to_csv('./assets/hw3/pA1.csv', index=None, float_format='%.3f')

fig, ax = plt.subplots()
ax.plot(xs, ys, '.')
plt.show()


N = 50
xs = np.random.rand(N) - 0.5
ys = -2 * xs ** 2 + 1 * xs + 4
noise = np.random.rand(N) * 0.05 - 0.025
ys = ys + noise

df = pd.DataFrame(columns=['x', 'y'])
df['x'] = xs
df['y'] = ys
df.to_csv('./assets/hw3/pA2.csv', index=None, float_format='%.3f')

fig, ax = plt.subplots()
ax.plot(xs, ys, '.')
plt.show()