import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


tau = 600
h = 0.1
x = np.arange(0, 10+h, h)
def exact_solution(x_i, t, a):
    u = np.zeros(len(x))
    for i in range(len(x)):
        u[i] = x[i] * x[i] * np.exp(-(x_i - x[i]) * (x_i - x[i]) / (4 * a * a * t))
    
    return np.trapz(u, x) / (2 * a * np.sqrt(np.pi * t))
    


def diagonal(A):
	C = make_coef(A)
	answ = list()
	answ.append(C[1][len(A) - 1])
	for i in range(2, len(A) + 1):
		answ.append(C[0][len(A) - i] * answ[i - 2] + C[1][len(A) - i])
	return answ


def make_coef(A):  
    v = 0.0
    C = list()
    alpha = list()
    beta = list()
    alpha.append(-A[0][2] / A[0][1])
    beta.append(A[0][3] / A[0][1])
    for i in range(1, len(A)):
        v = -A[i][1] - A[i][0] * alpha[i - 1]
        if i == len(A) - 1:
            alpha.append(0)
        else:
            alpha.append(A[i][2] / v)
     
        beta.append((A[i][0] * beta[i - 1] - A[i][3]) / v)
    
    C.append(alpha)
    C.append(beta)
    return C


def norma(u):
     u_0 = np.arange(0, 101)
     result = u_0 - u
     result = result ** 2
     s = np.sum(result)
     return np.sqrt(s)


f = open('data.txt','w')
u = np.zeros(2 * len(x)).reshape(2, len(x))
for i in range(2):
    u[i][100] = 100

a = 23e-6
plt.plot(x, u[0])
plt.title('Время: 0 мин')
plt.xlabel('x, м')
plt.ylabel('T, °C')
plt.savefig("./image/output0000.png")
number = 1
plt.close()
eps = 10
norm = 100
i = 0
while norm > eps:
    i += 1
    if i % 6 == 0:
         f.write(f"0 {i / 6} 0\n")
    step = len(x)
    coeff = tau * a / (h * h)
    A = np.zeros(4 * (step - 2)).reshape((step - 2), 4)
    A[0][1] = 1 + 2 * coeff
    A[0][2] = -coeff
    A[0][3] = u[i % 2 - 1][1] + coeff * u[i % 2][0]
    for j in range(1, step-3):
        A[j][0] = -coeff
        A[j][1] = 1 + 2 * coeff
        A[j][2] = -coeff
        A[j][3] = u[i % 2 - 1][j + 1]
        
    A[step-3][0] = -coeff
    A[step-3][1] = 1 + 2 * coeff
    A[step-3][3] = u[i % 2 - 1][step - 2] + coeff * u[i % 2][step - 1]
	
    answ = diagonal(A)
    for j in range(1, len(answ) + 1):
        u[i % 2][j] = answ[-j]
        if i % 6 == 0:
             f.write(f"{j * h} {i / 6} {u[i % 2][j]}\n")

    if i % 6 == 0:
         f.write(f"10 {i / 6} 100\n\n")
    norm = norma(u[i % 2])
    if i % 30 == 0:
        plt.plot(x, u[i % 2])
        plt.title(f"Время: {i / 6} часов")
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