import matplotlib.pyplot as pl
import numpy as np
def positive_phi(z):
    return -np.log(z)
def negative_phi(z):
    return -np.log(1-z)
z = np.arange(1e-5, 1, 0.01)
j_1 = positive_phi(z)
j_2 = negative_phi(z)
pl.plot(z, j_1, 'b-', label='J(w) if y=1')
pl.plot(z, j_2, 'g--', label='J(w) if y=0')
pl.ylim(0, 5.1)
pl.xlabel('$\phi(z)$')
pl.ylabel('$J(w)$')
pl.legend(loc='upper center')
pl.show()
