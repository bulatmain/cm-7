import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
a = 1.0          # коэффициент a, a^2 > 0
L = np.pi        # длина области [0, π]
T = 2.0          # конечное время

# Аналитическое решение
def analytical_solution(x, t):
    return np.sin(x - a*t) + np.cos(x + a*t)

# Начальные условия
def psi1(x):
    return np.sin(x) + np.cos(x)

def psi1_xx(x):
    return -psi1(x)

def psi2(x):
    return -a * (np.sin(x) + np.cos(x))

# Граничные условия (условия Робина)
# u_x(0,t) - u(0,t) = 0
# u_x(π,t) - u(π,t) = 0

# Явная схема крест
def explicit_cross_scheme(Nx, Nt, bc_approx='two_point_first', du_approx='first_order'):
    """
    Явная схема крест для волнового уравнения.
    
    Параметры:
    Nx - число узлов по пространству
    Nt - число шагов по времени
    bc_approx - метод аппроксимации граничных условий:
        'two_point_first' - двухточечная 1-го порядка
        'three_point_second' - трехточечная 2-го порядка
        'two_point_second' - двухточечная 2-го порядка
    du_approx - метод аппроксимации второго начального условия:
        'first_order' - первый порядок
        'second_order' - второй порядок
    """
    h = L / Nx
    tau = T / Nt
    
    # Коэффициент устойчивости Куранта
    c = a * tau / h
    if c > 1.0:
        print(f"Внимание: условие устойчивости может нарушаться (c={c:.3f} > 1)")
    
    # Сетки
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, T, Nt + 1)
    
    # Матрица решения (все временные слои)
    u = np.zeros((Nt + 1, Nx + 1))
    
    # Начальное условие: u(x,0)
    u[0, :] = psi1(x)
    
    # Второе начальное условие: u_t(x,0)
    # Создаем слой t=1 разными методами
    if du_approx == 'first_order':
        # Первый порядок: u_t ≈ (u^1 - u^0)/tau
        u[1, :] = u[0, :] + tau * psi2(x)
    else:  # second_order
        # Второй порядок: используем фиктивный слой u^{-1}
        # u_t ≈ (u^1 - u^{-1})/(2tau) = du0_dt
        # => u^{-1} = u^1 - 2tau*du0_dt
        # Подставляем в разностное уравнение при n=0
        for i in range(1, Nx):
            # Правая часть уравнения при n=0
            rhs = (a**2 * tau**2 / h**2) * (u[0, i+1] - 2*u[0, i] + u[0, i-1])
            # Из уравнения: u[1,i] - 2u[0,i] + u[-1,i] = rhs
            # u[-1,i] = u[1,i] - 2tau*psi2(x[i])
            # => u[1,i] - 2u[0,i] + u[1,i] - 2tau*psi2(x[i]) = rhs
            # => 2u[1,i] = rhs + 2u[0,i] + 2tau*psi2(x[i])
            u[1, i] = 0.5 * (rhs + 2*u[0, i] + 2*tau*psi2(x[i]))
        
        # Граничные условия для слоя t=1
        apply_boundary_conditions(u, 1, h, bc_approx)
    
    # Основной цикл по времени
    for n in range(1, Nt):
        # Внутренние точки
        for i in range(1, Nx):
            u[n+1, i] = (2*u[n, i] - u[n-1, i] + 
                        (a**2 * tau**2 / h**2) * (u[n, i+1] - 2*u[n, i] + u[n, i-1]))
        
        # Граничные условия
        apply_boundary_conditions(u, n+1, h, bc_approx)
    
    # Вычисление погрешности в последний момент времени
    u_analytical = analytical_solution(x, T)
    error_max = np.max(np.abs(u[-1, :] - u_analytical))
    error_l2 = np.sqrt(h * np.sum((u[-1, :] - u_analytical)**2))
    
    return x, u, {'h': h, 'tau': tau, 'c': c, 
                  'max_error': error_max, 'l2_error': error_l2}

def apply_boundary_conditions(u, time_layer, h, method):
    """
    Применение граничных условий Робина.
    u_x - u = 0 на обеих границах.
    """
    n = time_layer
    
    if method == 'two_point_first':
        # Левая граница: (u_1 - u_0)/h - u_0 = 0 => u_0 = u_1/(1+h)
        # Правая граница: (u_N - u_{N-1})/h - u_N = 0 => u_N = u_{N-1}/(1-h)
        u[n, 0] = u[n, 1] / (1 + h)
        u[n, -1] = u[n, -2] / (1 - h)
    
    elif method == 'three_point_second':
        # Левая граница: (-3u_0 + 4u_1 - u_2)/(2h) - u_0 = 0
        u[n, 0] = (4*u[n, 1] - u[n, 2]) / (3 + 2*h)
        # Правая граница: (3u_N - 4u_{N-1} + u_{N-2})/(2h) - u_N = 0
        u[n, -1] = (4*u[n, -2] - u[n, -3]) / (3 - 2*h)
    
    elif method == 'two_point_second':
        # Удалить этот метод или переименовать, т.к. он использует 3 точки
        # Оставляем для совместимости, но исправляем
        u[n, 0] = (4*u[n, 1] - u[n, 2]) / (3 + 2*h)
        u[n, -1] = (4*u[n, -2] - u[n, -3]) / (3 - 2*h)

# Неявная схема (с использованием метода Кранка-Николсон)
def implicit_scheme(Nx, Nt, bc_approx='two_point_first', du_approx='first_order'):
    h = L / Nx
    tau = T / Nt
    sigma = (a * tau / h)**2
    x = np.linspace(0, L, Nx + 1)
    u = np.zeros((Nt + 1, Nx + 1))
    u[0, :] = psi1(x)
    
    # Первый слой
    if du_approx == 'first_order':
        u[1, :] = u[0, :] + tau * psi2(x)
    elif du_approx == 'second_order':
        u[1, :] = u[0, :] + tau * psi2(x) + (tau**2 / 2.0) * a**2 * psi1_xx(x)
    
    # Применить граничные условия к первому слою (если нужно, но в неявной они встроены в матрицу для последующих)
    apply_boundary_conditions(u, 1, h, bc_approx)
    
    # Построение матрицы A
    A = np.zeros((Nx + 1, Nx + 1))
    for i in range(1, Nx):
        A[i, i-1] = -sigma / 2
        A[i, i] = 1 + sigma
        A[i, i+1] = -sigma / 2
    
    is_two_point_second = (bc_approx == 'two_point_second')
    
    if bc_approx == 'two_point_first':
        A[0, 0] = 1 + h
        A[0, 1] = -1
        A[Nx, Nx-1] = 1
        A[Nx, Nx] = -(1 - h)
    elif bc_approx == 'three_point_second' or is_two_point_second:  # Для three_point
        A[0, 0] = 3 + 2 * h
        A[0, 1] = -4
        A[0, 2] = 1
        A[Nx, Nx-2] = 1
        A[Nx, Nx-1] = -4
        A[Nx, Nx] = 3 - 2 * h
    if is_two_point_second:
        # Перезаписать для two_point_second
        A[0, :] = 0
        A[0, 0] = 1 + sigma * (1 + h)
        A[0, 1] = -sigma
        A[Nx, :] = 0
        A[Nx, Nx-1] = -sigma
        A[Nx, Nx] = 1 + sigma * (1 - h)
    
    # Основной цикл
    for n in range(1, Nt):
        b = np.zeros(Nx + 1)
        for i in range(1, Nx):
            b[i] = 2 * u[n, i] - u[n-1, i] + (sigma / 2) * (u[n-1, i-1] - 2 * u[n-1, i] + u[n-1, i+1])
        
        if bc_approx in ['two_point_first', 'three_point_second']:
            b[0] = 0
            b[Nx] = 0
        if is_two_point_second:
            # Special b for boundaries
            # Left
            delta_nm1_0 = (2 * u[n-1, 1] - 2 * (1 + h) * u[n-1, 0])
            b[0] = 2 * u[n, 0] - u[n-1, 0] + (sigma / 2) * delta_nm1_0
            # Right
            delta_nm1_N = (2 * u[n-1, Nx-1] - 2 * (1 - h) * u[n-1, Nx])
            b[Nx] = 2 * u[n, Nx] - u[n-1, Nx] + (sigma / 2) * delta_nm1_N
        
        u[n+1, :] = np.linalg.solve(A, b)
    
    # Погрешности
    u_analytical = analytical_solution(x, T)
    error_max = np.max(np.abs(u[-1, :] - u_analytical))
    error_l2 = np.sqrt(h * np.sum((u[-1, :] - u_analytical)**2))
    
    return x, u, {'h': h, 'tau': tau, 
                  'max_error': error_max, 'l2_error': error_l2}

def setup_boundary_matrix(A, b, index, h, method, u_prev):
    """
    Установка граничных условий в матрице для неявной схемы.
    """
    if method == 'two_point_first':
        if index == 0:  # левая граница
            A[index, index] = 1 + h
            A[index, index+1] = -1
            b[index] = 0
        else:  # правая граница
            A[index, index] = 1 + h
            A[index, index-1] = -1
            b[index] = 0
    
    elif method == 'three_point_second':
        if index == 0:  # левая граница
            A[index, index] = 3 + 2*h
            A[index, index+1] = -4
            A[index, index+2] = 1
            b[index] = 0
        else:  # правая граница
            A[index, index] = 3 + 2*h
            A[index, index-1] = -4
            A[index, index-2] = 1
            b[index] = 0
    
    elif method == 'two_point_second':
        if index == 0:  # левая граница
            A[index, index] = 3 + 2*h
            A[index, index+1] = -4
            A[index, index+2] = 1
            b[index] = 0
        else:  # правая граница
            A[index, index] = 3 + 2*h
            A[index, index-1] = -4
            A[index, index-2] = 1
            b[index] = 0

# Исследование зависимости погрешности от параметров сетки
def convergence_study(scheme_func, Nx_list, Nt_list, scheme_name):
    """
    Исследование сходимости схемы.
    """
    print(f"\nИсследование сходимости: {scheme_name}")
    
    errors_max = []
    errors_l2 = []
    h_values = []
    tau_values = []
    
    for Nx, Nt in zip(Nx_list, Nt_list):
        x, u, info = scheme_func(Nx, Nt)
        
        errors_max.append(info['max_error'])
        errors_l2.append(info['l2_error'])
        h_values.append(info['h'])
        tau_values.append(info['tau'])
        
        print(f"Nx={Nx}, Nt={Nt}: h={info['h']:.4f}, τ={info['tau']:.4f}")
        print(f"  Погрешность L∞: {info['max_error']:.2e}")
        print(f"  Погрешность L2: {info['l2_error']:.2e}")
        
        if 'c' in info:
            print(f"  Коэффициент Куранта: c={info['c']:.3f}")
    
    # Расчет порядка сходимости
    if len(h_values) >= 2:
        print("\nПорядок сходимости:")
        for i in range(1, len(h_values)):
            p_max = np.log(errors_max[i-1] / errors_max[i]) / np.log(h_values[i-1] / h_values[i])
            p_l2 = np.log(errors_l2[i-1] / errors_l2[i]) / np.log(h_values[i-1] / h_values[i])
            print(f"  h{i-1} -> h{i}: p_max={p_max:.3f}, p_l2={p_l2:.3f}")
    
    return h_values, errors_max, errors_l2

# Основная часть программы
def main():
    print(f"Параметры: a={a}, L=π, T={T}")
    print()
    
    # Базовые параметры сетки
    Nx = 200
    Nt = 800
    
    # Тестируем разные методы аппроксимации граничных условий
    bc_methods = ['two_point_first', 'three_point_second', 'two_point_second']
    du_methods = ['first_order', 'second_order']
    
    # 1. Явная схема с разными аппроксимациями
    print("\n1. Явная схема крест")
    print("=" * 40)
    
    for bc_method in bc_methods:
        for du_method in du_methods:
            print(f"\nГраничные: {bc_method}, Начальные: {du_method}")
            x, u, info = explicit_cross_scheme(Nx, Nt, bc_method, du_method)
            print(f"  Погрешность L∞: {info['max_error']:.2e}")
            print(f"  Погрешность L2: {info['l2_error']:.2e}")
            if 'c' in info:
                print(f"  Коэффициент Куранта: {info['c']:.3f}")
    
    # 2. Неявная схема
    print("\n\n2. Неявная схема")
    
    for bc_method in bc_methods:  # Теперь тестируем все методы
        for du_method in du_methods:
            print(f"\nГраничные: {bc_method}, Начальные: {du_method}")
            x, u, info = implicit_scheme(Nx, Nt, bc_method, du_method)
            print(f"  Погрешность L∞: {info['max_error']:.2e}")
            print(f"  Погрешность L2: {info['l2_error']:.2e}")
    
    # 3. Исследование сходимости
    print("\n\n3. Исследование сходимости")
    
    # Последовательности сеток для исследования сходимости
    Nx_list = [20, 40, 80, 160]
    Nt_list = [80, 160, 320, 640]  # сохраняем отношение τ/h
    
    # Явная схема с лучшими параметрами
    print("\nЯвная схема (трехточечные граничные, второй порядок по времени):")
    h_vals, err_max, err_l2 = convergence_study(
        lambda Nx, Nt: explicit_cross_scheme(Nx, Nt, 'three_point_second', 'second_order'),
        Nx_list, Nt_list, "Явная схема"
    )
    
    # Неявная схема
    print("\nНеявная схема (трехточечные граничные, второй порядок по времени):")
    h_vals_imp, err_max_imp, err_l2_imp = convergence_study(
        lambda Nx, Nt: implicit_scheme(Nx, Nt, 'three_point_second', 'second_order'),
        Nx_list, Nt_list, "Неявная схема"
    )
    
    # 4. Визуализация результатов
    print("\n\n4. Визуализация результатов")
    
    # Вычисляем решение для визуализации
    Nx_viz = 400
    Nt_viz = 400
    x, u_explicit, info_exp = explicit_cross_scheme(Nx_viz, Nt_viz, 'three_point_second', 'second_order')
    x, u_implicit, info_imp = implicit_scheme(Nx_viz, Nt_viz, 'three_point_second', 'second_order')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Моменты времени для визуализации
    time_indices = [0, Nt_viz//4, Nt_viz//2, 3*Nt_viz//4, Nt_viz]
    times = [i * (T / Nt_viz) for i in time_indices]
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    
    # График 1: Решение в разные моменты времени (явная схема)
    ax = axes[0, 0]
    for idx, t_idx in enumerate(time_indices):
        ax.plot(x, u_explicit[t_idx, :], color=colors[idx], 
                linewidth=2, label=f't={times[idx]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Явная схема: решение в разные моменты времени')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 2: Решение в разные моменты времени (неявная схема)
    ax = axes[0, 1]
    for idx, t_idx in enumerate(time_indices):
        ax.plot(x, u_implicit[t_idx, :], color=colors[idx], 
                linewidth=2, label=f't={times[idx]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Неявная схема: решение в разные моменты времени')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 3: Сравнение схем в конечный момент времени
    ax = axes[0, 2]
    ax.plot(x, u_explicit[-1, :], 'b-', linewidth=2, label='Явная схема')
    ax.plot(x, u_implicit[-1, :], 'r--', linewidth=2, label='Неявная схема')
    
    # Аналитическое решение
    u_analytical = analytical_solution(x, T)
    ax.plot(x, u_analytical, 'k:', linewidth=3, label='Аналитическое')
    
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,T)') 
    ax.set_title(f'Сравнение решений при t={T}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 4: Погрешности
    ax = axes[1, 0]
    error_exp = np.abs(u_explicit[-1, :] - u_analytical)
    error_imp = np.abs(u_implicit[-1, :] - u_analytical)
    
    ax.semilogy(x, error_exp, 'b-', linewidth=2, label='Явная схема')
    ax.semilogy(x, error_imp, 'r--', linewidth=2, label='Неявная схема')
    ax.set_xlabel('x')
    ax.set_ylabel('Абсолютная погрешность')
    ax.set_title('Погрешности в конечный момент времени')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 5: Сходимость явной схемы
    ax = axes[1, 1]
    ax.loglog(h_vals, err_max, 'bo-', linewidth=2, markersize=8, label='L∞ погрешность')
    ax.loglog(h_vals, err_l2, 'rs-', linewidth=2, markersize=8, label='L2 погрешность')
    
    # Линии для сравнения с теоретическим порядком
    h_test = np.array(h_vals)
    ax.loglog(h_test, 10*h_test**2, 'k--', linewidth=1, label='O(h²)')
    ax.loglog(h_test, 10*h_test, 'k:', linewidth=1, label='O(h)')
    
    ax.set_xlabel('Шаг по пространству h')
    ax.set_ylabel('Погрешность')
    ax.set_title('Сходимость явной схемы')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # График 6: Сходимость неявной схемы
    ax = axes[1, 2]
    ax.loglog(h_vals_imp, err_max_imp, 'bo-', linewidth=2, markersize=8, label='L∞ погрешность')
    ax.loglog(h_vals_imp, err_l2_imp, 'rs-', linewidth=2, markersize=8, label='L2 погрешность')
    
    ax.loglog(h_test, 10*h_test**2, 'k--', linewidth=1, label='O(h²)')
    ax.loglog(h_test, 10*h_test, 'k:', linewidth=1, label='O(h)')
    
    ax.set_xlabel('Шаг по пространству h')
    ax.set_ylabel('Погрешность')
    ax.set_title('Сходимость неявной схемы')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Исследование устойчивости явной схемы
    print("\n\n5. Исследование устойчивости явной схемы")
    
    Nx_test = 50
    c_values = [0.5, 0.8, 1.0, 1.2, 1.5]
    
    print("\nКоэффициент Куранта c = aτ/h")
    print("Условие устойчивости: c ≤ 1")
    print()
    
    for c in c_values:
        # Вычисляем τ из c = aτ/h
        h_test = L / Nx_test
        tau_test = c * h_test / a
        Nt_test = int(T / tau_test) + 1
        tau_test = T / Nt_test  # корректируем
        
        print(f"c = {c:.2f}: h = {h_test:.4f}, τ = {tau_test:.4f}, Nt = {Nt_test}")
        
        try:
            x, u, info = explicit_cross_scheme(Nx_test, Nt_test, 'three_point_second', 'second_order')
            actual_c = info.get('c', a * tau_test / h_test)
            print(f"  Фактический c: {actual_c:.3f}")
            print(f"  Погрешность: {info['max_error']:.2e}")
            
            # Проверка на неустойчивость
            max_val = np.max(np.abs(u))
            if max_val > 1e6:
                print(f"  Максимальное значение: {max_val:.2e}")
        except Exception as e:
            print(f"  Ошибка: {e}")

if __name__ == "__main__":
    main()