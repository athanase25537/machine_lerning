import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = x**2
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid()
plt.show()
