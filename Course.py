import numpy as np
import matplotlib.pyplot as plt

Num = 6
Npoint = 30
T0 = 1.874099616
y0 = 0.2
h = 2 * T0 / Npoint
x0 = 3
epsilon = 0.000001
delta = 1
count = 0
n = 5
m = n + 1

X = [i for i in range(0, Npoint)]
Y = [i for i in range(0, Npoint)]

A = np.zeros((n+1, n+1))
B = np.zeros(n+1)
S = np.zeros(2*n+1)

T = np.array([0, 0.1249, 0.2498, 0.3747, 0.4996, 0.6245, 0.7494, 0.8743, 0.9992, 1.1241, 1.2490, 1.3739, 1.4988, 1.6237, 1.7486, 1.8735,
              1.9984, 2.1233, 2.2482, 2.3731, 2.4980, 2.6229, 2.7478, 2.8727, 2.9976, 3.1225, 3.2474, 3.3723, 3.4972, 3.6221, 3.7470])
U = np.array([0.20000, 0.39474, 0.57187, 0.72189, 0.83832, 0.91818, 0.96200, 0.97331, 0.95777, 0.90925, 0.82637, 0.72264, 0.60966, 0.49637, 0.38898, 0.29118,
              0.37420, 0.42101, 0.44358, 0.45130, 0.45059, 0.44560, 0.43885, 0.43178, 0.41406, 0.38322, 0.34785, 0.31221, 0.27820, 0.24655, 0.21743])

# Метод касательных
def f(t):
    return float(t ** 5 - 7 * t - 10)

def kasat(t):
    return 5 * t ** 4 - 7

print("Метод касательных")
# Наименьший положительный корень T (Решение методом касательных)
while delta > epsilon:
    p = x0 - f(x0) / kasat(x0)
    delta = abs(p - x0)
    x0 = p
    print(count + 1, x0, p, delta)
    count = count + 1
    if count > 30:
        break
print('Наименьший положительный корень T =', x0)
x0 = round(x0, 5)
print('Сокращённый наименьший положительный корень T =', x0)

# Модельный вариант сигнала
def f16(x):
    if x < 0:
        return 0
    elif x <= 1:
        return 1
    elif x <= T0:
        return (x - T0) / (1 - T0)
    else:
        raise ValueError

# Сигнал
def Fs(t, T0):
    x = t - int(t / T0) * T0
    return f16(x)

# Численное интегрирование
def Fun(t, y):
    return 0.1 * t ** 2 - 2 * t * y + (1 + Num / 10) * Fs(t, T0)

# Метод Эйлера второго порядка с коррекцией по средней производной
def Euler_Mid_deriv(f, t0, y0, h, N):
    a = 0
    b = 0
    X[-1] = t0
    Y[-1] = y0
    t = t0
    y = y0
    for j in range(1, N + 1):
        z = f(t, y)
        yz = y + h * z
        z1 = f(t + h, yz)
        y = y + h * (z + z1) / 0.2e1
        t = t + h
        X[j - 1] = t
        Y[j - 1] = y
        a = str(round(t, 4))
        b = str(round(y, 6))
        print(a, b)
        file.write(a + ' ' + b + '\n')

file = open('Course.txt', 'w+')
Euler_Mid_deriv(Fun, 0, y0, h, Npoint)
file.close()

def cholesky(A):
    # Разложение Холецкого для симметричной положительно определённой матрицы A.
    # Возвращает матрицу L такую, что A = L*L.T.
    n = len(A)
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i+1):
            if i == j:
                # Вычисление диагональных элементов
                L[i][j] = np.sqrt(max(A[i][i] - np.sum(L[i][:i]**2), 0))
            else:
                # Вычисление элементов вне диагонали
                L[i][j] = (A[i][j] - np.sum(L[i][:j]*L[j][:j])) / L[j][j]
    
    return L

for j in range(Npoint+1):
    p = 1
    for k in range(2*n+1):
        S[k] += p
        p *= T[j]

    p = 1
    for k in range(n+1):
        B[k] += U[j] * p
        p *= T[j]

for j in range(n+1):
    for k in range(n+1):
        A[j, k] = S[j+k]

print("Матрица A")
print(A)
print("Вектор B")
print(B)

np.savetxt('A.txt', A, fmt='%.4f', delimiter='\n')
np.savetxt('B.txt', B, fmt='%.4f', delimiter='\n')

L = cholesky(A)
print("Метод Холецкого")
print(L)

# Построение графика матрицы L
plt.imshow(L, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.show()