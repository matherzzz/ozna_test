import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

f = open('data.txt','w')
tau = 0.01
h = 0.01
x = np.arange(0, 1+h, h)
t = np.arange(0, 1+tau, tau)

u = np.zeros(101 * len(x)).reshape(101, len(x))
for i in range(101):
    u[i][100] = 100

a = 23e-6
plt.plot(x, u[0])
plt.title('Время: 0 мин')
plt.xlabel('x, м')
plt.ylabel('T, °C')
plt.savefig("./image/output0000.png")
number = 1
plt.close()
for j in range(1, len(x)):
    f.write(f"{j * h} 0 0\n")

for i in range(1, (len(t)- 1) * 18000 + 1):
    if i % 1800 == 0:
        f.write(f"0 {i / 6000} 0\n")
    for j in range(1, len(x) - 1):
        u[i % 101][j] = u[i % 101-1][j] + tau * a * (u[i % 101-1][j+1] - 2 * u[i % 101-1][j] + u[i % 101-1][j-1]) / (h * h)
        if i % 1800 == 0:
            f.write(f"{j * h} {i / 6000} {u[i % 101][j]}\n")
    
    if i % 1800 == 0:
        f.write(f"1 {i / 6000} 100\n\n")
    if i % 30000 == 0:
        plt.plot(x, u[0])
        plt.title(f"Время: {i / 6000} мин")
        plt.xlabel('x, м')
        plt.ylabel('T, °C')
        plt.savefig("./image/output" + '0' * (4-len(str(number))) + str(number)  +".png")
        number += 1
        plt.close()

png_dir = './image'
images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))

# Make it pause at the end so that the viewers can ponder
for _ in range(10):
    images.append(imageio.imread(file_path))

imageio.mimsave('./animation/movie.gif', images, fps = 10, loop = 5)
f.close()