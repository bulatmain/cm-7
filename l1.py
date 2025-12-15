import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from typing import Tuple, Dict, List, Callable

# Параметры
a = 1.0
L = 1.0
T = 0.5

# Граничные условия
def left_boundary(t):
    return 0.0

def right_boundary(t):
    return 1.0

# Начальное условие
def initial_condition(x):
    return x + np.sin(np.pi * x)

# Аналитическое решение
def analytical_solution(x, t):
    return x + np.exp(-np.pi**2 * a * t) * np.sin(np.pi * x)

# Вспомогательные функции
def create_grid(Nx: int, Nt: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Создает равномерную пространственно-временную сетку.
    
    Возвращает:
    x - массив пространственных узлов
    t - массив временных узлов
    h - шаг по пространству
    tau - шаг по времени
    """
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, T, Nt + 1)
    h = x[1] - x[0]
    tau = t[1] - t[0]
    return x, t, h, tau

def tridiagonal_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Решение трехдиагональной системы методом прогонки (алгоритм Томаса).
    
    Система имеет вид:
    b[0]*u[0] + c[0]*u[1] = d[0]
    a[i]*u[i-1] + b[i]*u[i] + c[i]*u[i+1] = d[i], i=1..N-2
    a[N-1]*u[N-2] + b[N-1]*u[N-1] = d[N-1]
    """
    n = len(d)
    
    # Прямой ход прогонки
    alpha = np.zeros(n)
    beta = np.zeros(n)
    
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denominator = b[i] + a[i] * alpha[i-1]
        alpha[i] = -c[i] / denominator
        beta[i] = (d[i] - a[i] * beta[i-1]) / denominator
    
    # Обратный ход прогонки
    u = np.zeros(n)
    denominator = b[n-1] + a[n-1] * alpha[n-2]
    u[n-1] = (d[n-1] - a[n-1] * beta[n-2]) / denominator
    
    for i in range(n-2, -1, -1):
        u[i] = alpha[i] * u[i+1] + beta[i]
    
    return u

def compute_errors(u_numerical: np.ndarray, u_analytical: np.ndarray, h: float) -> Dict[str, float]:
    """
    Вычисляет различные нормы погрешности.
    """
    error = u_numerical - u_analytical
    
    max_error = np.max(np.abs(error))
    
    l2_error = np.sqrt(h * np.sum(error[1:-1]**2))
    
    return {
        'max': max_error,
        'l2': l2_error
    }

def explicit_scheme(Nx: int, Nt: int, save_all_steps: bool = False) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Явная конечно-разностная схема.
    
    Возвращает:
    u - матрица решения (если save_all_steps=True) или последний временной слой
    x - пространственные узлы
    tau - шаг по времени
    info - информация о вычислениях
    """
    x, t, h, tau = create_grid(Nx, Nt)
    
    # Проверка устойчивости
    r = a * tau / h**2
    if r > 0.5:
        print(f"Условие устойчивости не выполняется: r = {r:.3f} > 0.5")
    
    # Инициализация решения
    if save_all_steps:
        u = np.zeros((Nt + 1, Nx + 1))
        u[0, :] = initial_condition(x)
    else:
        u_prev = initial_condition(x)
        u_curr = np.zeros_like(u_prev, dtype=np.float128)
    
    # Временной цикл
    start_time = time.time()
    
    if save_all_steps:
        for n in range(Nt):
            u[n+1, 0] = left_boundary(t[n+1])
            u[n+1, Nx] = right_boundary(t[n+1])
            
            for i in range(1, Nx):
                u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        u_final = u[-1, :]
    else:
        u_prev = initial_condition(x)
        for n in range(Nt):
            u_curr[0] = left_boundary(t[n+1])
            u_curr[Nx] = right_boundary(t[n+1])
            
            for i in range(1, Nx):
                u_curr[i] = u_prev[i] + r * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1])
            
            u_prev = u_curr.copy()
        u_final = u_curr
    
    elapsed_time = time.time() - start_time
    
    # Вычисление погрешности
    u_analytical = analytical_solution(x, T)
    errors = compute_errors(u_final, u_analytical, h)
    
    info = {
        'h': h,
        'tau': tau,
        'r': r,
        'time': elapsed_time,
        'errors': errors,
        'scheme': 'Явная схема'
    }
    
    if save_all_steps:
        return u, x, t, info
    else:
        return u_final, x, info

def implicit_scheme(Nx: int, Nt: int, save_all_steps: bool = False) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Неявная конечно-разностная схема.
    """
    x, t, h, tau = create_grid(Nx, Nt)
    r = a * tau / h**2
    
    # Инициализация
    if save_all_steps:
        u = np.zeros((Nt + 1, Nx + 1))
        u[0, :] = initial_condition(x)
    else:
        u_prev = initial_condition(x)
    
    # Коэффициенты для трехдиагональной системы
    n_inner = Nx - 1  # внутренние точки
    A = np.zeros(n_inner)  # нижняя диагональ
    B = np.zeros(n_inner)  # главная диагональ  
    C = np.zeros(n_inner)  # верхняя диагональ
    D = np.zeros(n_inner)  # правая часть
    
    # Заполняем постоянные коэффициенты
    A[:] = -r
    B[:] = 1 + 2*r
    C[:] = -r
    
    start_time = time.time()
    
    if save_all_steps:
        for n in range(Nt):
            # Граничные условия
            u[n+1, 0] = left_boundary(t[n+1])
            u[n+1, Nx] = right_boundary(t[n+1])
            
            # Правая часть
            for i in range(1, Nx):
                D[i-1] = u[n, i]
            
            # Учет граничных условий в правой части
            D[0] += r * left_boundary(t[n+1])
            D[-1] += r * right_boundary(t[n+1])
            
            # Решение системы
            u_inner = tridiagonal_solve(A, B, C, D)
            u[n+1, 1:-1] = u_inner
        
        u_final = u[-1, :]
    else:
        u_prev = initial_condition(x)
        u_curr = np.zeros_like(u_prev)
        
        for n in range(Nt):
            u_curr[0] = left_boundary(t[n+1])
            u_curr[Nx] = right_boundary(t[n+1])
            
            # Правая часть
            for i in range(1, Nx):
                D[i-1] = u_prev[i]
            
            # Учет граничных условий
            D[0] += r * left_boundary(t[n+1])
            D[-1] += r * right_boundary(t[n+1])
            
            # Решение системы
            u_inner = tridiagonal_solve(A, B, C, D)
            u_curr[1:-1] = u_inner
            
            u_prev = u_curr.copy()
        
        u_final = u_curr
    
    elapsed_time = time.time() - start_time
    
    # Погрешность
    u_analytical = analytical_solution(x, T)
    errors = compute_errors(u_final, u_analytical, h)
    
    info = {
        'h': h,
        'tau': tau,
        'r': r,
        'time': elapsed_time,
        'errors': errors,
        'scheme': 'Неявная схема'
    }
    
    if save_all_steps:
        return u, x, t, info
    else:
        return u_final, x, info

def crank_nicolson_scheme(Nx: int, Nt: int, save_all_steps: bool = False) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Схема Кранка-Николсона.
    """
    x, t, h, tau = create_grid(Nx, Nt)
    r = a * tau / (2 * h**2)  # половинный коэффициент для CN
    
    # Инициализация
    if save_all_steps:
        u = np.zeros((Nt + 1, Nx + 1))
        u[0, :] = initial_condition(x)
    else:
        u_prev = initial_condition(x)
    
    # Коэффициенты для трехдиагональной системы
    n_inner = Nx - 1
    A = np.zeros(n_inner)  # нижняя диагональ
    B = np.zeros(n_inner)  # главная диагональ
    C = np.zeros(n_inner)  # верхняя диагональ
    D = np.zeros(n_inner)  # правая часть
    
    # Заполняем постоянные коэффициенты
    A[:] = -r
    B[:] = 1 + 2*r
    C[:] = -r
    
    start_time = time.time()
    
    if save_all_steps:
        for n in range(Nt):
            u[n+1, 0] = left_boundary(t[n+1])
            u[n+1, Nx] = right_boundary(t[n+1])
            
            # Вычисление правой части
            for i in range(1, Nx):
                # Явная часть
                explicit_part = r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
                D[i-1] = u[n, i] + explicit_part
            
            # Учет граничных условий в правой части
            D[0] += r * left_boundary(t[n+1])
            D[-1] += r * right_boundary(t[n+1])
            
            # Решение системы
            u_inner = tridiagonal_solve(A, B, C, D)
            u[n+1, 1:-1] = u_inner
        
        u_final = u[-1, :]
    else:
        u_prev = initial_condition(x)
        u_curr = np.zeros_like(u_prev)
        
        for n in range(Nt):
            u_curr[0] = left_boundary(t[n+1])
            u_curr[Nx] = right_boundary(t[n+1])
            
            # Правая часть
            for i in range(1, Nx):
                explicit_part = r * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1])
                D[i-1] = u_prev[i] + explicit_part
            
            # Учет граничных условий
            D[0] += r * left_boundary(t[n+1])
            D[-1] += r * right_boundary(t[n+1])
            
            # Решение системы
            u_inner = tridiagonal_solve(A, B, C, D)
            u_curr[1:-1] = u_inner
            
            u_prev = u_curr.copy()
        
        u_final = u_curr
    
    elapsed_time = time.time() - start_time
    
    # Погрешность
    u_analytical = analytical_solution(x, T)
    errors = compute_errors(u_final, u_analytical, h)
    
    info = {
        'h': h,
        'tau': tau,
        'r': r,
        'time': elapsed_time,
        'errors': errors,
        'scheme': 'Схема Кранка-Николсона'
    }
    
    if save_all_steps:
        return u, x, t, info
    else:
        return u_final, x, info

# Исследование сходимости
def convergence_study(scheme_func: Callable, Nx_list: List[int], Nt_list: List[int], 
                     scheme_name: str = "") -> Dict:
    """
    Исследование сходимости схемы при различных сетках.
    
    Args:
        scheme_func: функция схемы (explicit_scheme, implicit_scheme, crank_nicolson_scheme)
        Nx_list: список значений Nx
        Nt_list: список значений Nt (должен быть той же длины, что и Nx_list)
        scheme_name: название схемы для вывода
    
    Returns:
        Словарь с результатами исследования
    """
    print(f"Исследование сходимости: {scheme_name}")
    
    results = {
        'Nx': [],
        'Nt': [],
        'h': [],
        'tau': [],
        'max_error': [],
        'l2_error': [],
        'computation_time': []
    }
    
    for Nx, Nt in zip(Nx_list, Nt_list):
        print(f"\nСетка: Nx = {Nx}, Nt = {Nt}")
        
        # Выполнение расчета
        u_numerical, x, info = scheme_func(Nx, Nt, save_all_steps=False)
        
        # Сохранение результатов
        results['Nx'].append(Nx)
        results['Nt'].append(Nt)
        results['h'].append(info['h'])
        results['tau'].append(info['tau'])
        results['max_error'].append(info['errors']['max'])
        results['l2_error'].append(info['errors']['l2'])
        results['computation_time'].append(info['time'])
        
        print(f"  h = {info['h']:.6f}, τ = {info['tau']:.6f}")
        print(f"  r = aτ/h² = {info['r']:.6f}")
        print(f"  Время расчета: {info['time']:.4f} с")
        print(f"  Макс. погрешность: {info['errors']['max']:.6e}")
        print(f"  L2-погрешность: {info['errors']['l2']:.6e}")
    
    # Расчет порядка сходимости
    if len(Nx_list) >= 2:
        print(f"\nОценка порядка сходимости:")
        
        # По пространству (при фиксированном отношении τ/h²)
        for i in range(1, len(Nx_list)):
            h_ratio = results['h'][i-1] / results['h'][i]
            
            # Порядок по максимальной погрешности
            if results['max_error'][i] > 0 and results['max_error'][i-1] > 0:
                p_max = np.log(results['max_error'][i-1] / results['max_error'][i]) / np.log(h_ratio)
            else:
                p_max = np.nan
            
            # Порядок по L2-погрешности
            if results['l2_error'][i] > 0 and results['l2_error'][i-1] > 0:
                p_l2 = np.log(results['l2_error'][i-1] / results['l2_error'][i]) / np.log(h_ratio)
            else:
                p_l2 = np.nan
            
            print(f"  От h={results['h'][i-1]:.4f} до h={results['h'][i]:.4f}:")
            print(f"    Порядок сходимости (L∞): {p_max:.4f}")
            print(f"    Порядок сходимости (L2): {p_l2:.4f}")
    
    return results

def plot_solutions(u_explicit, u_implicit, u_cn, u_analytical, x, scheme_names=None):
    """
    Построение графиков решений.
    """
    if scheme_names is None:
        scheme_names = ['Явная схема', 'Неявная схема', 'Кранка-Николсона', 'Аналитическое']
    
    plt.figure(figsize=(12, 8))
    
    # Графики решений
    plt.subplot(2, 2, 1)
    plt.plot(x, u_explicit, 'b-', linewidth=2, label=scheme_names[0])
    plt.plot(x, u_implicit, 'r--', linewidth=2, label=scheme_names[1])
    plt.plot(x, u_cn, 'g-.', linewidth=2, label=scheme_names[2])
    plt.plot(x, u_analytical, 'k:', linewidth=3, label=scheme_names[3])
    plt.xlabel('x')
    plt.ylabel('u(x,T)')
    plt.title(f'Сравнение численных решений при t={T}')
    plt.legend()
    plt.grid(True)
    
    # Погрешности
    plt.subplot(2, 2, 2)
    plt.plot(x, np.abs(u_explicit - u_analytical), 'b-', linewidth=2, label=scheme_names[0])
    plt.plot(x, np.abs(u_implicit - u_analytical), 'r--', linewidth=2, label=scheme_names[1])
    plt.plot(x, np.abs(u_cn - u_analytical), 'g-.', linewidth=2, label=scheme_names[2])
    plt.xlabel('x')
    plt.ylabel('Абсолютная погрешность')
    plt.title('Погрешности численных решений')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Начальное условие и аналитическое решение в разные моменты времени
    plt.subplot(2, 2, 3)
    t_values = [0, T/4, T/2, 3*T/4, T]
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for t_val, color in zip(t_values, colors):
        if t_val == 0:
            u = initial_condition(x)
            label = f't = {t_val:.3f} (начальное)'
        else:
            u = analytical_solution(x, t_val)
            label = f't = {t_val:.3f}'
        plt.plot(x, u, color=color, linewidth=2, label=label)
    
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Аналитическое решение в разные моменты времени')
    plt.legend()
    plt.grid(True)
    
    # 3D-график временной эволюции (если есть данные)
    plt.subplot(2, 2, 4, projection='3d')
    
    # Для 3D графика возьмем более грубую сетку
    Nx_3d, Nt_3d = 40, 50
    u_3d, x_3d, t_3d, _ = crank_nicolson_scheme(Nx_3d, Nt_3d, save_all_steps=True)
    
    # Создаем сетку для 3D
    X, T_mesh = np.meshgrid(x_3d, t_3d)
    
    # Построение поверхности
    surf = plt.gca().plot_surface(X, T_mesh, u_3d, cmap=cm.viridis, 
                                 linewidth=0, antialiased=True, alpha=0.8)
    
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('t')
    plt.gca().set_zlabel('u(x,t)')
    plt.title('Временная эволюция (схема Кранка-Николсона)')
    plt.colorbar(surf, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plt.show()

def plot_convergence(results_dict, scheme_names=None):
    """
    Построение графиков сходимости.
    """
    if scheme_names is None:
        scheme_names = list(results_dict.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: Погрешность L∞ от h
    ax = axes[0, 0]
    for name in scheme_names:
        if name in results_dict:
            results = results_dict[name]
            ax.loglog(results['h'], results['max_error'], 'o-', linewidth=2, 
                     markersize=8, label=name)
    
    # Теоретические линии сходимости
    h_min = min([min(r['h']) for r in results_dict.values()])
    h_max = max([max(r['h']) for r in results_dict.values()])
    h_test = np.logspace(np.log10(h_min), np.log10(h_max), 10)
    
    # O(h²) и O(τ) линии для сравнения
    ax.loglog(h_test, 10*h_test**2, 'k--', linewidth=1, label='O(h²)')
    ax.loglog(h_test, 10*h_test, 'k:', linewidth=1, label='O(h)')
    
    ax.set_xlabel('Шаг по пространству h')
    ax.set_ylabel('Максимальная погрешность')
    ax.set_title('Сходимость по пространству (L∞-норма)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # График 2: Погрешность L2 от h
    ax = axes[0, 1]
    for name in scheme_names:
        if name in results_dict:
            results = results_dict[name]
            ax.loglog(results['h'], results['l2_error'], 's-', linewidth=2, 
                     markersize=8, label=name)
    
    ax.loglog(h_test, 10*h_test**2, 'k--', linewidth=1, label='O(h²)')
    ax.loglog(h_test, 10*h_test, 'k:', linewidth=1, label='O(h)')
    
    ax.set_xlabel('Шаг по пространству h')
    ax.set_ylabel('L2-погрешность')
    ax.set_title('Сходимость по пространству (L2-норма)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # График 3: Время расчета от числа узлов
    ax = axes[1, 0]
    for name in scheme_names:
        if name in results_dict:
            results = results_dict[name]
            total_nodes = [nx*nt for nx, nt in zip(results['Nx'], results['Nt'])]
            ax.loglog(total_nodes, results['computation_time'], 'd-', 
                     linewidth=2, markersize=8, label=name)
    
    ax.set_xlabel('Общее число узлов (Nx × Nt)')
    ax.set_ylabel('Время расчета (с)')
    ax.set_title('Зависимость времени расчета от размера сетки')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # График 4: Отношение погрешностей для оценки порядка
    ax = axes[1, 1]
    if len(scheme_names) > 0 and scheme_names[0] in results_dict:
        results = results_dict[scheme_names[0]]
        if len(results['h']) >= 2:
            h_ratios = []
            error_ratios_max = []
            error_ratios_l2 = []
            
            for i in range(1, len(results['h'])):
                h_ratio = results['h'][i-1] / results['h'][i]
                if results['max_error'][i] > 0:
                    error_ratio_max = results['max_error'][i-1] / results['max_error'][i]
                    error_ratio_l2 = results['l2_error'][i-1] / results['l2_error'][i]
                    
                    h_ratios.append(h_ratio)
                    error_ratios_max.append(error_ratio_max)
                    error_ratios_l2.append(error_ratio_l2)
            
            if h_ratios:
                ax.plot(h_ratios, error_ratios_max, 'o-', linewidth=3, 
                       markersize=10, label='L∞-погрешность')
                ax.plot(h_ratios, error_ratios_l2, 's-', linewidth=2, 
                       markersize=10, label='L2-погрешность')
                
                # Линия для O(h²): error_ratio ≈ h_ratio²
                x_line = np.array([min(h_ratios), max(h_ratios)])
                ax.plot(x_line, x_line**2, 'k--', linewidth=2, label='O(h²)')
                
                ax.set_xlabel('Отношение шагов h_{i-1}/h_i')
                ax.set_ylabel('Отношение погрешностей E_{i-1}/E_i')
                ax.set_title('Оценка порядка сходимости')
                ax.legend()
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# Аппроксимация
def apply_boundary_conditions(u, h, bc_type, bc_value, method='two_point_first'):
    """
    Применение граничных условий с производными.
    
    Args:
        u: массив решения
        h: шаг по пространству
        bc_type: тип границы ('left' или 'right')
        bc_value: значение производной на границе
        method: метод аппроксимации:
            'two_point_first' - двухточечная первого порядка
            'three_point_second' - трехточечная второго порядка
            'two_point_second' - двухточечная второго порядка
    """
    if method == 'two_point_first':
        # Двухточечная аппроксимация первого порядка
        if bc_type == 'left':
            # (u1 - u0)/h = bc_value
            u[0] = u[1] - h * bc_value
        else:  # right
            # (u_N - u_{N-1})/h = bc_value
            u[-1] = u[-2] + h * bc_value
    
    elif method == 'three_point_second':
        # Трехточечная аппроксимация второго порядка
        if bc_type == 'left':
            # (-3u0 + 4u1 - u2)/(2h) = bc_value
            u[0] = (4*u[1] - u[2] - 2*h*bc_value) / 3
        else:  # right
            # (3u_N - 4u_{N-1} + u_{N-2})/(2h) = bc_value
            u[-1] = (4*u[-2] - u[-3] + 2*h*bc_value) / 3
    
    elif method == 'two_point_second':
        # Двухточечная аппроксимация второго порядка (асимметричная)
        if bc_type == 'left':
            # (-3u0 + 4u1 - u2)/(2h) = bc_value (та же, что и трехточечная)
            u[0] = (4*u[1] - u[2] - 2*h*bc_value) / 3
        else:  # right
            # (u_{N-2} - 4u_{N-1} + 3u_N)/(2h) = bc_value
            u[-1] = (4*u[-2] - u[-3] + 2*h*bc_value) / 3
    
    return u

def main():
    # Параметры для базового расчета
    Nx_base, Nt_base = 50, 500
    
    print(f"\nБазовый расчет на сетке Nx={Nx_base}, Nt={Nt_base}")
    print(f"Коэффициент a={a}")
    print(f"Конечное время T={T}")
    
    # 1. Выполнение расчетов тремя методами
    print("\n1. Расчет тремя методами:")
    
    # Явная схема
    print("\n  Явная схема:")
    u_explicit, x_explicit, info_explicit = explicit_scheme(Nx_base, Nt_base)
    print(f"    Время: {info_explicit['time']:.4f} с")
    print(f"    Макс. погрешность: {info_explicit['errors']['max']:.6e}")
    print(f"    L2-погрешность: {info_explicit['errors']['l2']:.6e}")
    
    # Неявная схема
    print("\n  Неявная схема:")
    u_implicit, x_implicit, info_implicit = implicit_scheme(Nx_base, Nt_base)
    print(f"    Время: {info_implicit['time']:.4f} с")
    print(f"    Макс. погрешность: {info_implicit['errors']['max']:.6e}")
    print(f"    L2-погрешность: {info_implicit['errors']['l2']:.6e}")
    
    # Схема Кранка-Николсона
    print("\n  Схема Кранка-Николсона:")
    u_cn, x_cn, info_cn = crank_nicolson_scheme(Nx_base, Nt_base)
    print(f"    Время: {info_cn['time']:.4f} с")
    print(f"    Макс. погрешность: {info_cn['errors']['max']:.6e}")
    print(f"    L2-погрешность: {info_cn['errors']['l2']:.6e}")
    
    # Аналитическое решение
    u_analytical = analytical_solution(x_explicit, T)
    
    # 2. Визуализация результатов
    print("\n2. Построение графиков...")
    plot_solutions(u_explicit, u_implicit, u_cn, u_analytical, x_explicit)
    
    # 3. Исследование сходимости
    print("\n3. Исследование сходимости схем...")
    
    # Определяем последовательности сеток для исследования сходимости
    # Для сохранения отношения τ/h² (особенно важно для явной схемы)
    Nx_list = [10, 20, 40, 80, 160]
    Nt_list = [200, 400, 800, 1600, 3200]  # Увеличиваем Nt для сохранения τ/h²
    
    # Исследование для каждой схемы
    results_explicit = convergence_study(explicit_scheme, Nx_list, Nt_list, "Явная схема")
    results_implicit = convergence_study(implicit_scheme, Nx_list, Nt_list, "Неявная схема")
    results_cn = convergence_study(crank_nicolson_scheme, Nx_list, Nt_list, "Схема Кранка-Николсона")
    
    # Объединяем результаты для визуализации
    results_dict = {
        'Явная схема': results_explicit,
        'Неявная схема': results_implicit,
        'Схема Кранка-Николсона': results_cn
    }
    
    # Графики сходимости
    plot_convergence(results_dict)
    
    # 4. Демонстрация аппроксимации граничных условий с производными
    print("\n4. Демонстрация аппроксимации граничных условий с производными")
    print("   (на примере фиктивного решения)")
    
    # Создаем тестовый массив
    N_test = 10
    x_test = np.linspace(0, 1, N_test)
    u_test = np.sin(np.pi * x_test)  # Тестовое решение
    
    # Значение производной на границе (произвольное)
    bc_value = 1.0
    h_test = x_test[1] - x_test[0]
    
    print(f"\n   Исходное решение на левой границе: u[0] = {u_test[0]:.6f}")
    print(f"   Требуемое значение производной: u'(0) = {bc_value}")
    
    # Применяем разные методы аппроксимации
    methods = [
        ('Двухточечная 1-го порядка', 'two_point_first'),
        ('Трехточечная 2-го порядка', 'three_point_second'),
        ('Двухточечная 2-го порядка', 'two_point_second')
    ]
    
    for method_name, method_code in methods:
        u_copy = u_test.copy()
        u_modified = apply_boundary_conditions(u_copy, h_test, 'left', bc_value, method_code)
        print(f"\n   {method_name}:")
        print(f"     Новое значение u[0] = {u_modified[0]:.6f}")
        # Вычисляем аппроксимацию производной для проверки
        if method_code == 'two_point_first':
            approx_deriv = (u_modified[1] - u_modified[0]) / h_test
        elif method_code in ['three_point_second', 'two_point_second']:
            approx_deriv = (-3*u_modified[0] + 4*u_modified[1] - u_modified[2]) / (2*h_test)
        print(f"     Вычисленная производная: {approx_deriv:.6f}")
        print(f"     Погрешность: {abs(approx_deriv - bc_value):.6e}")
    
    # Сводная таблица результатов
    print("\nСводная таблица результатов (базовый расчет):")
    print("-"*80)
    print(f"{'Схема':<25} {'Время (с)':<12} {'L∞-погрешность':<20} {'L2-погрешность':<20}")
    print("-"*80)
    print(f"{'Явная':<25} {info_explicit['time']:<12.4f} {info_explicit['errors']['max']:<20.2e} {info_explicit['errors']['l2']:<20.2e}")
    print(f"{'Неявная':<25} {info_implicit['time']:<12.4f} {info_implicit['errors']['max']:<20.2e} {info_implicit['errors']['l2']:<20.2e}")
    print(f"{'Кранка-Николсона':<25} {info_cn['time']:<12.4f} {info_cn['errors']['max']:<20.2e} {info_cn['errors']['l2']:<20.2e}")
    print("-"*80)

# Запуск основной программы
if __name__ == "__main__":
    main()