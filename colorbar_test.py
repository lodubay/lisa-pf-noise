import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(1000, 2000, 50)
f = np.linspace(0, 1, 1000)
d = np.random.normal(size=(len(f), len(t)))

data = np.array([
    np.array(sorted(list(t) * len(f))),
    np.array(list(f) * len(t)),
    np.concatenate(d)
])

plt.imshow(d, cmap='coolwarm', 
        #aspect=len(t)/len(f), 
        aspect='auto',
        origin='lower',
        extent=[min(t), max(t), min(f), max(f)])
plt.xlabel('GPS time at start')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.show()
