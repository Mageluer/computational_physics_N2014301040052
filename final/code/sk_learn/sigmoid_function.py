import matplotlib.pyplot as pl
import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
pl.plot(z, phi_z)
pl.axvline(0.0, color='k')
pl.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
pl.axhline(y=0.5, ls='dotted', color='k')
pl.yticks([0.0, 0.5, 1.0])
pl.ylim(-0.1, 1.1)
pl.xlabel('z')
pl.ylabel('$\phi (z)$')
pl.title('sigmoid function')
pl.show()
