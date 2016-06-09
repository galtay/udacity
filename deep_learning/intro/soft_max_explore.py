"""
softmax turns scores into probabilities

S(y_i) = e^{y_i} / sum_{k=1}^{k=N} for
"""

## example of softmax
import numpy as np
import matplotlib.pyplot as plt

scores = [3.0, 1.0, 0.2]

def softmax(x):
    """Compute softmax values for x."""
    xarr = np.array(x)
    e_to_the_x = np.exp(xarr)
    sm = e_to_the_x / e_to_the_x.sum(axis=0)
    return sm


print(softmax(scores))

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
