import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 25})


loss = np.load('Loss1.npy')

plt.figure(figsize=(15, 6))
plt.plot(loss[0:2000, 0], label='Total loss')
plt.plot(loss[0:2000, 1], label='Content loss')
plt.plot(loss[0:2000, 2], label='Style loss')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('Loss_zoom_in.png')

