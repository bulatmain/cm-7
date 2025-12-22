import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Параметры задачи
Lx, Ly = 1.0, 1.0  # область [0,1]x[0,1]

# Аналитическое решение
def analytical_solution(x, y):
    return x**2 - y**2

# Граничные условия
def set_boundary_conditions(u, hx, hy, Nx, Ny, bc_type='three_point'):
    """
    Установка граничных условий.
    bc_type: 'two_point_first', 'three_point', 'two_point_second'
    """
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    
    # Правая граница: u(1, y) = 1 - y^2 (Дирихле)
    for j in range(Ny+1):
        u[Nx, j] = 1 - y[j]**2
    
    # Верхняя граница: u(x, 1) = x^2 - 1 (Дирихле)
    for i in range(Nx+1):
        u[i, Ny] = x[i]**2 - 1
    
    # Левая граница: u_x(0, y) = 0 (Нейман)
    # Используем трехточечную аппроксимацию второго порядка
    if bc_type == 'three_point':
        for j in range(1, Ny):
            # (-3u0,j + 4u1,j - u2,j) / (2hx) = 0
            # u0,j = (4u1,j - u2,j) / 3
            u[0, j] = (4*u[1, j] - u[2, j]) / 3
    elif bc_type == 'two_point_first':
        # (u1,j - u0,j) / hx = 0 => u0,j = u1,j
        for j in range(1, Ny):
            u[0, j] = u[1, j]
    
    # Нижняя граница: u_y(x, 0) = 0 (Нейман)
    if bc_type == 'three_point':
        for i in range(1, Nx):
            # (-3ui,0 + 4ui,1 - ui,2) / (2hy) = 0
            # ui,0 = (4ui,1 - ui,2) / 3
            u[i, 0] = (4*u[i, 1] - u[i, 2]) / 3
    elif bc_type == 'two_point_first':
        # (ui,1 - ui,0) / hy = 0 => ui,0 = ui,1
        for i in range(1, Nx):
            u[i, 0] = u[i, 1]
    
    # Угловые точки
    # Левый нижний угол (0,0) - оба условия Неймана
    if bc_type == 'three_point':
        # Используем усреднение из обоих условий
        u[0, 0] = 0.5 * ((4*u[1, 0] - u[2, 0])/3 + (4*u[0, 1] - u[0, 2])/3)
    elif bc_type == 'two_point_first':
        u[0, 0] = 0.5 * (u[1, 0] + u[0, 1])
    
    # Левый верхний угол (0,Ny) - Нейман по x, Дирихле по y
    # Значение уже установлено из условия Дирихле на верхней границе
    
    # Правый нижний угол (Nx,0) - Дирихле по x, Нейман по y
    # Значение уже установлено из условия Дирихле на правой границе
    
    return u

# Метод простых итераций (Якоби/Либман)
def jacobi_method(Nx, Ny, max_iter=10000, tol=1e-6, omega=1.0, bc_type='three_point'):
    """
    Метод простых итераций (с возможностью релаксации).
    omega=1.0 - обычный метод Якоби
    omega>1.0 - верхняя релаксация
    omega<1.0 - нижняя релаксация
    """
    hx, hy = Lx/Nx, Ly/Ny
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    
    # Инициализация
    u = np.zeros((Nx+1, Ny+1))
    u_new = np.zeros((Nx+1, Ny+1))
    
    # Установка граничных условий
    u = set_boundary_conditions(u, hx, hy, Nx, Ny, bc_type)
    u_new = u.copy()
    
    # Коэффициенты
    coeff_x = 1/hx**2
    coeff_y = 1/hy**2
    denom = 2*(coeff_x + coeff_y)
    
    # Итерационный процесс
    errors = []
    for k in range(max_iter):
        max_change = 0.0
        
        # Обновление внутренних точек
        for i in range(1, Nx):
            for j in range(1, Ny):
                # Старое значение по Якоби
                u_jacobi = (coeff_x*(u[i-1, j] + u[i+1, j]) + 
                           coeff_y*(u[i, j-1] + u[i, j+1])) / denom
                # Релаксация
                u_new[i, j] = (1 - omega)*u[i, j] + omega*u_jacobi
                max_change = max(max_change, abs(u_new[i, j] - u[i, j]))
        
        # Обновление границ Неймана
        if bc_type == 'three_point':
            # Левая граница (кроме углов)
            for j in range(1, Ny):
                u_new[0, j] = (4*u_new[1, j] - u_new[2, j]) / 3
            
            # Нижняя граница (кроме углов)
            for i in range(1, Nx):
                u_new[i, 0] = (4*u_new[i, 1] - u_new[i, 2]) / 3
            
            # Левый нижний угол
            u_new[0, 0] = 0.5 * ((4*u_new[1, 0] - u_new[2, 0])/3 + 
                                (4*u_new[0, 1] - u_new[0, 2])/3)
        
        elif bc_type == 'two_point_first':
            for j in range(1, Ny):
                u_new[0, j] = u_new[1, j]
            for i in range(1, Nx):
                u_new[i, 0] = u_new[i, 1]
            u_new[0, 0] = 0.5 * (u_new[1, 0] + u_new[0, 1])
        
        # Копируем новое решение
        u[:, :] = u_new[:, :]
        
        errors.append(max_change)
        if max_change < tol:
            print(f"Метод Якоби сошелся за {k+1} итераций")
            break
    
    # Вычисление погрешности
    u_analytical = analytical_solution(x[:, None], y[None, :])
    error_max = np.max(np.abs(u - u_analytical))
    error_l2 = np.sqrt(hx*hy * np.sum((u - u_analytical)**2))
    
    return u, error_max, error_l2, errors

# Метод Зейделя
def seidel_method(Nx, Ny, max_iter=10000, tol=1e-6, bc_type='three_point'):
    """
    Метод Зейделя (последовательное обновление).
    """
    hx, hy = Lx/Nx, Ly/Ny
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    
    # Инициализация
    u = np.zeros((Nx+1, Ny+1))
    u = set_boundary_conditions(u, hx, hy, Nx, Ny, bc_type)
    
    # Коэффициенты
    coeff_x = 1/hx**2
    coeff_y = 1/hy**2
    denom = 2*(coeff_x + coeff_y)
    
    errors = []
    for k in range(max_iter):
        max_change = 0.0
        
        # Обновление в порядке: снизу вверх, слева направо
        for j in range(1, Ny):
            for i in range(1, Nx):
                old_val = u[i, j]
                # Используем уже обновленные значения
                u[i, j] = (coeff_x*(u[i-1, j] + u[i+1, j]) + 
                          coeff_y*(u[i, j-1] + u[i, j+1])) / denom
                max_change = max(max_change, abs(u[i, j] - old_val))
        
        # Обновление границ Неймана
        if bc_type == 'three_point':
            for j in range(1, Ny):
                u[0, j] = (4*u[1, j] - u[2, j]) / 3
            for i in range(1, Nx):
                u[i, 0] = (4*u[i, 1] - u[i, 2]) / 3
            u[0, 0] = 0.5 * ((4*u[1, 0] - u[2, 0])/3 + (4*u[0, 1] - u[0, 2])/3)
        
        elif bc_type == 'two_point_first':
            for j in range(1, Ny):
                u[0, j] = u[1, j]
            for i in range(1, Nx):
                u[i, 0] = u[i, 1]
            u[0, 0] = 0.5 * (u[1, 0] + u[0, 1])
        
        errors.append(max_change)
        if max_change < tol:
            print(f"Метод Зейделя сошелся за {k+1} итераций")
            break
    
    # Вычисление погрешности
    u_analytical = analytical_solution(x[:, None], y[None, :])
    error_max = np.max(np.abs(u - u_analytical))
    error_l2 = np.sqrt(hx*hy * np.sum((u - u_analytical)**2))
    
    return u, error_max, error_l2, errors

# Метод верхней релаксации (SOR)
def sor_method(Nx, Ny, omega=1.5, max_iter=10000, tol=1e-6, bc_type='three_point'):
    """
    Метод верхней релаксации (Successive Over-Relaxation).
    """
    hx, hy = Lx/Nx, Ly/Ny
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    
    # Инициализация
    u = np.zeros((Nx+1, Ny+1))
    u = set_boundary_conditions(u, hx, hy, Nx, Ny, bc_type)
    
    # Коэффициенты
    coeff_x = 1/hx**2
    coeff_y = 1/hy**2
    denom = 2*(coeff_x + coeff_y)
    
    errors = []
    for k in range(max_iter):
        max_change = 0.0
        
        # Обновление с релаксацией
        for j in range(1, Ny):
            for i in range(1, Nx):
                old_val = u[i, j]
                # Значение по Гауссу-Зейделю
                u_gs = (coeff_x*(u[i-1, j] + u[i+1, j]) + 
                       coeff_y*(u[i, j-1] + u[i, j+1])) / denom
                # Релаксация
                u[i, j] = (1 - omega)*old_val + omega*u_gs
                max_change = max(max_change, abs(u[i, j] - old_val))
        
        # Обновление границ Неймана
        if bc_type == 'three_point':
            for j in range(1, Ny):
                u[0, j] = (4*u[1, j] - u[2, j]) / 3
            for i in range(1, Nx):
                u[i, 0] = (4*u[i, 1] - u[i, 2]) / 3
            u[0, 0] = 0.5 * ((4*u[1, 0] - u[2, 0])/3 + (4*u[0, 1] - u[0, 2])/3)
        
        elif bc_type == 'two_point_first':
            for j in range(1, Ny):
                u[0, j] = u[1, j]
            for i in range(1, Nx):
                u[i, 0] = u[i, 1]
            u[0, 0] = 0.5 * (u[1, 0] + u[0, 1])
        
        errors.append(max_change)
        if max_change < tol:
            print(f"Метод SOR (ω={omega}) сошелся за {k+1} итераций")
            break
    
    # Вычисление погрешности
    u_analytical = analytical_solution(x[:, None], y[None, :])
    error_max = np.max(np.abs(u - u_analytical))
    error_l2 = np.sqrt(hx*hy * np.sum((u - u_analytical)**2))
    
    return u, error_max, error_l2, errors

# Исследование сходимости
def convergence_study(method_func, N_values, method_name, **kwargs):
    """
    Исследование зависимости погрешности от шага сетки.
    """
    print(f"\n{method_name}:")
    print("-" * 50)
    
    errors_max = []
    errors_l2 = []
    h_values = []
    iterations_list = []
    
    for N in N_values:
        Nx, Ny = N, N  # квадратная сетка
        h = Lx / N
        
        u, error_max, error_l2, errors = method_func(Nx, Ny, **kwargs)
        
        errors_max.append(error_max)
        errors_l2.append(error_l2)
        h_values.append(h)
        iterations_list.append(len(errors))
        
        print(f"N={N}, h={h:.4f}: max error={error_max:.2e}, L2 error={error_l2:.2e}, iterations={len(errors)}")
    
    # Расчет порядка сходимости
    if len(h_values) >= 2:
        print("\nПорядок сходимости:")
        for i in range(1, len(h_values)):
            p_max = np.log(errors_max[i-1] / errors_max[i]) / np.log(h_values[i-1] / h_values[i])
            p_l2 = np.log(errors_l2[i-1] / errors_l2[i]) / np.log(h_values[i-1] / h_values[i])
            print(f"h{i-1}->h{i}: p_max={p_max:.3f}, p_l2={p_l2:.3f}")
    
    return h_values, errors_max, errors_l2, iterations_list

# Визуализация результатов
def plot_results(Nx, Ny):
    """
    Визуализация численного и аналитического решений.
    """
    hx, hy = Lx/Nx, Ly/Ny
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    
    # Вычисляем решения разными методами
    u_jacobi, _, _, _ = jacobi_method(Nx, Ny, max_iter=5000, tol=1e-6)
    u_seidel, _, _, _ = seidel_method(Nx, Ny, max_iter=5000, tol=1e-6)
    u_sor, _, _, _ = sor_method(Nx, Ny, omega=1.8, max_iter=5000, tol=1e-6)
    u_analytical = analytical_solution(x[:, None], y[None, :])
    
    # Создаем сетку для 3D графиков
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    fig = plt.figure(figsize=(15, 10))
    
    # График 1: Аналитическое решение
    ax1 = fig.add_subplot(231, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_analytical, cmap='viridis', alpha=0.8)
    ax1.set_title('Аналитическое решение: u(x,y)=x²-y²')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # График 2: Метод Якоби
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_jacobi, cmap='plasma', alpha=0.8)
    ax2.set_title('Метод Якоби')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # График 3: Метод Зейделя
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(X, Y, u_seidel, cmap='coolwarm', alpha=0.8)
    ax3.set_title('Метод Зейделя')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('u(x,y)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # График 4: Метод SOR
    ax4 = fig.add_subplot(234, projection='3d')
    surf4 = ax4.plot_surface(X, Y, u_sor, cmap='summer', alpha=0.8)
    ax4.set_title('Метод SOR (ω=1.8)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('u(x,y)')
    fig.colorbar(surf4, ax=ax4, shrink=0.5)
    
    # График 5: Погрешности
    ax5 = fig.add_subplot(235)
    
    error_jacobi = np.abs(u_jacobi - u_analytical)
    error_seidel = np.abs(u_seidel - u_analytical)
    error_sor = np.abs(u_sor - u_analytical)
    
    # Средняя погрешность по x для фиксированного y (середина области)
    j_mid = Ny // 2
    ax5.plot(x, error_jacobi[:, j_mid], 'b-', linewidth=2, label='Якоби')
    ax5.plot(x, error_seidel[:, j_mid], 'r--', linewidth=2, label='Зейделя')
    ax5.plot(x, error_sor[:, j_mid], 'g-.', linewidth=2, label='SOR')
    
    ax5.set_xlabel('x')
    ax5.set_ylabel('Абсолютная погрешность')
    ax5.set_title(f'Погрешности при y={y[j_mid]:.2f}')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # График 6: Сходимость итераций
    ax6 = fig.add_subplot(236)
    
    # Вычисляем ошибки по итерациям для сетки 20x20
    N_test = 20
    _, _, _, errors_j = jacobi_method(N_test, N_test, max_iter=200, tol=1e-10)
    _, _, _, errors_s = seidel_method(N_test, N_test, max_iter=200, tol=1e-10)
    _, _, _, errors_sor = sor_method(N_test, N_test, omega=1.8, max_iter=200, tol=1e-10)
    
    ax6.semilogy(range(1, len(errors_j)+1), errors_j, 'b-', linewidth=2, label='Якоби')
    ax6.semilogy(range(1, len(errors_s)+1), errors_s, 'r--', linewidth=2, label='Зейделя')
    ax6.semilogy(range(1, len(errors_sor)+1), errors_sor, 'g-.', linewidth=2, label='SOR (ω=1.8)')
    
    ax6.set_xlabel('Номер итерации')
    ax6.set_ylabel('Максимальное изменение')
    ax6.set_title('Сходимость итерационных методов')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Сравнение методов аппроксимации граничных условий
def compare_boundary_approximations(Nx, Ny):
    """
    Сравнение разных методов аппроксимации граничных условий.
    """
    print("\nСравнение методов аппроксимации граничных условий:")
    print("=" * 60)
    
    methods = [
        ('Двухточечная 1-го порядка', 'two_point_first'),
        ('Трехточечная 2-го порядка', 'three_point'),
    ]
    
    results = {}
    
    for method_name, bc_type in methods:
        print(f"\n{method_name}:")
        
        # Метод Якоби
        u_j, e_max_j, e_l2_j, iter_j = jacobi_method(Nx, Ny, bc_type=bc_type, max_iter=5000)
        print(f"  Якоби: max error={e_max_j:.2e}, L2 error={e_l2_j:.2e}, iterations={len(iter_j)}")
        
        # Метод Зейделя
        u_s, e_max_s, e_l2_s, iter_s = seidel_method(Nx, Ny, bc_type=bc_type, max_iter=5000)
        print(f"  Зейделя: max error={e_max_s:.2e}, L2 error={e_l2_s:.2e}, iterations={len(iter_s)}")
        
        # Метод SOR
        u_sor, e_max_sor, e_l2_sor, iter_sor = sor_method(Nx, Ny, omega=1.8, bc_type=bc_type, max_iter=5000)
        print(f"  SOR: max error={e_max_sor:.2e}, L2 error={e_l2_sor:.2e}, iterations={len(iter_sor)}")
        
        results[method_name] = {
            'jacobi': (e_max_j, e_l2_j, len(iter_j)),
            'seidel': (e_max_s, e_l2_s, len(iter_s)),
            'sor': (e_max_sor, e_l2_sor, len(iter_sor))
        }
    
    return results

# Основная программа
def main():
    print("Лабораторная работа 3: Уравнение Лапласа")
    print("Вариант 2: ∂²u/∂x² + ∂²u/∂y² = 0")
    print("Граничные условия:")
    print("  u_x(0, y) = 0")
    print("  u(1, y) = 1 - y²")
    print("  u_y(x, 0) = 0")
    print("  u(x, 1) = x² - 1")
    print("Аналитическое решение: u(x, y) = x² - y²")
    print()
    
    # Параметры для исследования
    N_values = [10, 20]  # размеры сеток для исследования сходимости
    N_viz = 30  # размер сетки для визуализации
    
    # 1. Сравнение методов аппроксимации граничных условий
    print("\n1. Сравнение методов аппроксимации граничных условий")
    results_bc = compare_boundary_approximations(20, 20)
    
    # 2. Исследование сходимости по h
    print("\n\n2. Исследование сходимости")
    
    # Метод Якоби
    h_j, e_max_j, e_l2_j, iter_j = convergence_study(
        jacobi_method, N_values, "Метод Якоби (трехточечные границы)"
    )
    
    # Метод Зейделя
    h_s, e_max_s, e_l2_s, iter_s = convergence_study(
        seidel_method, N_values, "Метод Зейделя (трехточечные границы)"
    )
    
    # Метод SOR
    h_sor, e_max_sor, e_l2_sor, iter_sor = convergence_study(
        lambda Nx, Ny: sor_method(Nx, Ny, omega=1.8),
        N_values, "Метод SOR с ω=1.8 (трехточечные границы)"
    )
    
    # 3. Визуализация
    print("\n\n3. Визуализация результатов")
    plot_results(N_viz, N_viz)
    
    # 4. Дополнительный анализ
    print("\n\n4. Дополнительный анализ")
    
    # Оптимальный параметр релаксации для SOR
    print("\nПоиск оптимального параметра релаксации ω для SOR:")
    
    omega_values = np.linspace(1.0, 2.0, 11)
    iterations = []
    errors = []
    
    for omega in omega_values:
        _, _, _, errors_iter = sor_method(20, 20, omega=omega, max_iter=1000, tol=1e-6)
        iterations.append(len(errors_iter))
        errors.append(errors_iter[-1] if len(errors_iter) < 1000 else float('inf'))
    
    # Построение графика зависимости числа итераций от ω
    plt.figure(figsize=(10, 6))
    plt.plot(omega_values, iterations, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Параметр релаксации ω')
    plt.ylabel('Число итераций до сходимости')
    plt.title('Зависимость скорости сходимости SOR от параметра ω')
    plt.grid(True, alpha=0.3)
    
    # Находим оптимальное ω
    opt_idx = np.argmin(iterations)
    opt_omega = omega_values[opt_idx]
    print(f"Оптимальное ω ≈ {opt_omega:.2f} (итераций: {iterations[opt_idx]})")
    
    plt.axvline(x=opt_omega, color='r', linestyle='--', alpha=0.7, 
                label=f'Оптимальное ω={opt_omega:.2f}')
    plt.legend()
    plt.show()
    
    # 5. Сводная таблица результатов
    print("\n\n5. Сводная таблица результатов (N=20)")
    print("{:<15} {:<12} {:<12} {:<12}".format(
        "Метод", "L∞-ошибка", "L2-ошибка", "Итерации"))
    
    # Вычисляем для N=20
    N = 20
    u_j, e_max_j, e_l2_j, iter_j = jacobi_method(N, N, max_iter=5000)
    u_s, e_max_s, e_l2_s, iter_s = seidel_method(N, N, max_iter=5000)
    u_sor, e_max_sor, e_l2_sor, iter_sor = sor_method(N, N, omega=opt_omega, max_iter=5000)
    
    print("{:<15} {:<12.2e} {:<12.2e} {:<12}".format(
        "Якоби", e_max_j, e_l2_j, len(iter_j)))
    print("{:<15} {:<12.2e} {:<12.2e} {:<12}".format(
        "Зейделя", e_max_s, e_l2_s, len(iter_s)))
    print("{:<15} {:<12.2e} {:<12.2e} {:<12}".format(
        f"SOR (ω={opt_omega:.1f})", e_max_sor, e_l2_sor, len(iter_sor)))

if __name__ == "__main__":
    main()