import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Параметры задачи
a = 1.0
Lx, Ly = np.pi, np.pi  # область
T = 1.0                # конечное время

# Метод прогонки (TDMA)
def tdma(a, b, c, d):
    n = len(d)
    cp, dp = np.zeros(n), np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = 1.0 / (b[i] - a[i]*cp[i-1])
        cp[i] = c[i] * m
        dp[i] = (d[i] - a[i]*dp[i-1]) * m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

# Аналитическое решение
def analytic_solution(x, y, t, m1, m2):
    return np.cos(m1*x) * np.cos(m2*y) * np.exp(-(m1**2 + m2**2)*a*t)

# Решение для одного случая
def solve_case(m1, m2, Nx=50, Ny=50, dt_val=0.001, save_plots=True):
    # Сетка
    dx, dy = Lx/(Nx-1), Ly/(Ny-1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    
    Nt = int(T/dt_val) + 1
    
    # Инициализация
    u = np.zeros((Nx, Ny))
    u_new = np.zeros((Nx, Ny))
    
    # Начальное условие
    for i in range(Nx):
        for j in range(Ny):
            u[i,j] = np.cos(m1*x[i]) * np.cos(m2*y[j])
    
    # Коэффициенты для прогонки
    alpha = a * dt_val / (2 * dx**2)
    beta = a * dt_val / (2 * dy**2)
    
    # Массивы для прогонки по x
    A_x = -alpha * np.ones(Nx-2)
    B_x = (1 + 2*alpha) * np.ones(Nx-2)
    C_x = -alpha * np.ones(Nx-2)
    
    # Массивы для прогонки по y
    A_y = -beta * np.ones(Ny-2)
    B_y = (1 + 2*beta) * np.ones(Ny-2)
    C_y = -beta * np.ones(Ny-2)
    
    errors = []
    time_steps = []
    
    # Временной цикл
    for n in range(Nt):
        t = n * dt_val
        
        # Шаг 1: половинный шаг по x
        for j in range(1, Ny-1):
            d = np.zeros(Nx-2)
            for i in range(1, Nx-1):
                d[i-1] = u[i,j] + beta*(u[i,j-1] - 2*u[i,j] + u[i,j+1])
            # Граничные условия по x для шага 1
            d[0] += alpha * np.cos(m2*y[j]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val/2))
            d[-1] += alpha * ((-1)**m1) * np.cos(m2*y[j]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val/2))
            u_temp = tdma(A_x, B_x, C_x, d)
            u_new[1:-1, j] = u_temp
        
        # Граничные условия после шага 1
        for j in range(Ny):
            u_new[0,j] = np.cos(m2*y[j]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val/2))
            u_new[-1,j] = ((-1)**m1) * np.cos(m2*y[j]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val/2))
        for i in range(Nx):
            u_new[i,0] = np.cos(m1*x[i]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val/2))
            u_new[i,-1] = ((-1)**m2) * np.cos(m1*x[i]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val/2))
        
        # Шаг 2: половинный шаг по y
        for i in range(1, Nx-1):
            d = np.zeros(Ny-2)
            for j in range(1, Ny-1):
                d[j-1] = u_new[i,j] + alpha*(u_new[i-1,j] - 2*u_new[i,j] + u_new[i+1,j])
            # Граничные условия по y для шага 2
            d[0] += beta * np.cos(m1*x[i]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val))
            d[-1] += beta * ((-1)**m2) * np.cos(m1*x[i]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val))
            u_temp = tdma(A_y, B_y, C_y, d)
            u[i, 1:-1] = u_temp
        
        # Граничные условия после шага 2
        for j in range(Ny):
            u[0,j] = np.cos(m2*y[j]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val))
            u[-1,j] = ((-1)**m1) * np.cos(m2*y[j]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val))
        for i in range(Nx):
            u[i,0] = np.cos(m1*x[i]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val))
            u[i,-1] = ((-1)**m2) * np.cos(m1*x[i]) * np.exp(-(m1**2 + m2**2)*a*(t+dt_val))
        
        # Сбор ошибок в определённые моменты времени
        if n % max(1, Nt//4) == 0:
            u_anal = analytic_solution(X, Y, t, m1, m2)
            error = np.abs(u.T - u_anal).max()
            errors.append(error)
            time_steps.append(t)
            
            # Визуализация в момент времени t
            if save_plots and n % (max(1, Nt//2)) == 0:  # реже строим графики
                fig = plt.figure(figsize=(15,5))
                # Численное решение
                ax1 = fig.add_subplot(131, projection='3d')
                ax1.plot_surface(X, Y, u.T, cmap='viridis')
                ax1.set_title(f'Численное, t={t:.2f}')
                # Аналитическое решение
                ax2 = fig.add_subplot(132, projection='3d')
                ax2.plot_surface(X, Y, u_anal, cmap='plasma')
                ax2.set_title('Аналитическое')
                # Погрешность
                ax3 = fig.add_subplot(133, projection='3d')
                ax3.plot_surface(X, Y, np.abs(u.T - u_anal), cmap='hot')
                ax3.set_title('Погрешность')
                plt.suptitle(f'm1={m1}, m2={m2}, t={t:.2f}')
                plt.tight_layout()
                plt.show()
    
    # График зависимости ошибки от времени
    plt.figure(figsize=(8,5))
    plt.plot(time_steps, errors, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Время')
    plt.ylabel('Максимальная погрешность')
    plt.title(f'Погрешность от времени (m1={m1}, m2={m2})')
    plt.grid(True)
    plt.show()
    
    # Исследование зависимости от шагов сетки
    if save_plots:
        N_list = [20, 30, 50, 70]
        errors_h = []
        dt_list = []
        for N in N_list:
            dx_local = Lx/(N-1)
            dy_local = Ly/(N-1)
            dt_local = min(dx_local**2, dy_local**2)/4
            dt_list.append(dt_local)
            # Сокращенный расчет для оценки ошибки
            u_test, err_test = solve_case(m1, m2, N, N, dt_local, save_plots=False)
            errors_h.append(err_test)
            
        plt.figure(figsize=(8,5))
        plt.loglog(N_list, errors_h, marker='s', linestyle='--', linewidth=2)
        plt.xlabel('N (число узлов)')
        plt.ylabel('Погрешность')
        plt.title('Зависимость погрешности от размера сетки')
        plt.grid(True)
        plt.show()
    
    return u, errors[-1] if errors else 0.0

# Запуск для трех случаев
cases = [(1,1), (2,1), (1,2)]
results = []
for m1, m2 in cases:
    print(f'Решается случай m1={m1}, m2={m2}...')
    u_final, err = solve_case(m1, m2, save_plots=True)
    results.append((m1, m2, err))
    print(f'Погрешность в конечный момент: {err:.2e}\n')