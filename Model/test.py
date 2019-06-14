
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
N_points = 100000
n_bins = 20

x = np.log(np.random.randn(N_points) + 10)


plt.hist(x, bins=n_bins)

plt.show()