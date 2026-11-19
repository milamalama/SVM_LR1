import numpy as np

#Исходные данные
A = np.array([
    [10.64, 2.13, -1.47, 0.82],
    [1.72, 13.25, 1.16, -2.08],
    [-2.14, 1.36, 11.83, 1.57],
    [1.25, -1.48, 2.06, 12.74]])

b = np.array([14.36, 18.71, 16.52, 20.18])
n = len(b)
eps = 1e-6
max_iter = 1000

print("Матрица A:")
print(A)
print("Вектор b:", b)
print("Определитель:", np.linalg.det(A))
print("Число обусловленности:", np.linalg.cond(A))

#Библиотечное решение
x_lib = np.linalg.solve(A, b)
print("\nБиблиотечное решение:", x_lib)

#Метод Гаусса
def gauss(A, b):
    A = A.copy()
    b = b.copy()
    n = len(b)
    
    for k in range(n):
        p = k + np.argmax(np.abs(A[k:, k]))
        
        if A[p, k] == 0:
            raise ValueError("Матрица вырождена")
            
        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]
        
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i] = A[i] - m * A[k]
            b[i] = b[i] - m * b[k]
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        s = 0
        
        for j in range(i + 1, n):
            s += A[i, j] * x[j]
        x[i] = (b[i] - s) / A[i, i]
        
    return x

#Метод простой итерации
def jacobi(A, b, x0, eps, max_iter):
    n = len(b)
    x = x0.copy()
    
    for k in range(1, max_iter + 1):
        x_new = np.zeros(n)
        
        for i in range(n):
            s = 0
            
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
                    
            x_new[i] = (b[i] - s) / A[i, i]
            
        error = np.linalg.norm(x_new - x)
        print("Итерация:", k,"Приближение:", x_new,"Изменение:", error)
        
        if error < eps:
            return x_new, k
        x = x_new
        
    print("Требуемая точность не достигнута")
    return x, max_iter

#Метод Зейделя
def seidel(A, b, x0, eps, max_iter):
    n = len(b)
    x = x0.copy()
    
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        
        for i in range(n):
            s = 0
            
            for j in range(n):
                if j != i:
                    
                    s += A[i, j] * x[j]
            x[i] = (b[i] - s) / A[i, i]
        error = np.linalg.norm(x - x_old)
        
        print("Итерация:", k,"Приближение:", x,"Изменение:", error)
        if error < eps:
            return x, k
            
    print("Требуемая точность не достигнута")
    return x, max_iter

#Вычисление нормы невязки
def residual(A, b, x):
    return np.linalg.norm(A @ x - b)

#Отклонение от библиотечного решения
def deviation(x, x_lib):
    return np.linalg.norm(x - x_lib)

#Решение системы всеми методами
print("\nМетод Гаусса")
x_gauss = gauss(A, b)
print("Решение:", x_gauss)

print("\nМетод простой итерации")
x_jacobi, k_jacobi = jacobi(A, b, np.zeros(n), eps, max_iter)
print("Решение:", x_jacobi)
print("Количество итераций:", k_jacobi)

print("\nМетод Зейделя")
x_seidel, k_seidel = seidel(A, b, np.zeros(n), eps, max_iter)
print("Решение:", x_seidel)
print("Количество итераций:", k_seidel)


#Сравнение результатов
print("\nСравнение методов")
methods = {"Гаусс": x_gauss,"Простая итерация": x_jacobi,"Зейдель": x_seidel}

for name, x in methods.items():
    print("\nМетод:", name)
    print("Решение:", x)
    print("Норма невязки:", residual(A, b, x))
    print("Отклонение от NumPy:", deviation(x, x_lib))

#Исследование влияния начального приближения
print("\nИсследование начальных приближений")
initial_vectors = [np.zeros(n),np.ones(n),np.array([5, -2, 3, 1], dtype=float)]

for x0 in initial_vectors:
    print("\nНачальное приближение:", x0)
    
    print("\nМетод простой итерации")
    x, k = jacobi(A, b, x0, eps, max_iter)
    print("Решение:", x)
    print("Количество итераций:", k)
    print("Норма невязки:", residual(A, b, x))
    
    print("\nМетод Зейделя")
    x, k = seidel(A, b, x0, eps, max_iter)
    print("Решение:", x)
    print("Количество итераций:", k)
    print("Норма невязки:", residual(A, b, x))

#Исследование влияния заданной точности
print("\nИсследование заданной точности")

for e in [1e-2, 1e-4, 1e-6]:
    print("\nТочность:", e)
    
    print("\nМетод простой итерации")
    x, k = jacobi(A, b, np.zeros(n), e, max_iter)
    print("Решение:", x)
    print("Количество итераций:", k)
    print("Норма невязки:", residual(A, b, x))
    
    print("\nМетод Зейделя")
    x, k = seidel(A, b, np.zeros(n), e, max_iter)
    print("Решение:", x)
    print("Количество итераций:", k)
    print("Норма невязки:", residual(A, b, x))

#Исследование влияния округления коэффициентов
print("\nИсследование округления коэффициентов")

for digits in [0, 1, 2]:
    A_round = np.round(A, digits)
    x_round = gauss(A_round, b)
    
    print("\nКоличество знаков после запятой:", digits)
    print("Округленная матрица:")
    print(A_round)
    print("Решение:", x_round)
    print("Изменение решения:", deviation(x_round, x_lib))
