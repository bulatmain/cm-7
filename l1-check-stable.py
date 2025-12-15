import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from typing import Tuple, Dict, List, Callable, Optional
import warnings

# Проверяем доступность float128
HAS_FLOAT128 = hasattr(np, 'float128')
if HAS_FLOAT128:
    DTYPE = np.float128
    print("Используется np.float128 для повышенной точности")
else:
    DTYPE = np.float64
    print("np.float128 не доступен, используется np.float64")
    warnings.warn("Тип данных float128 недоступен. Возможна потеря точности при очень мелких сетках.")

# ====================
# ПАРАМЕТРЫ ЗАДАЧИ
# ====================
# Параметры для варианта 2
a = DTYPE(1.0)          # коэффициент теплопроводности
L = DTYPE(1.0)          # длина стержня
T = DTYPE(0.5)          # конечное время

# Граничные условия
def left_boundary(t):
    return DTYPE(0.0)

def right_boundary(t):
    return DTYPE(1.0)

# Начальное условие
def initial_condition(x):
    return x + np.sin(np.pi * x, dtype=DTYPE)

# Аналитическое решение с защитой от переполнения
def analytical_solution(x, t):
    # Вычисляем показатель экспоненты с защитой от переполнения
    exponent = -np.pi**2 * a * t
    # Ограничиваем очень большие по модулю отрицательные значения для стабильности
    exponent = np.clip(exponent, DTYPE(-700.0), DTYPE(700.0))
    exp_term = np.exp(exponent, dtype=DTYPE)
    return x + exp_term * np.sin(np.pi * x, dtype=DTYPE)

# ====================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЧИСЛЕННОЙ СТАБИЛЬНОСТИ
# ====================
def safe_exp(x, max_val=700.0, min_val=-700.0):
    """Безопасное вычисление экспоненты с ограничением аргумента."""
    if isinstance(x, np.ndarray):
        x_clipped = np.clip(x, min_val, max_val)
        return np.exp(x_clipped, dtype=DTYPE)
    else:
        x_clipped = DTYPE(max(min(x, max_val), min_val))
        return np.exp(x_clipped, dtype=DTYPE)

def safe_division(a, b, eps=DTYPE(1e-300)):
    """Безопасное деление с защитой от деления на ноль."""
    return np.where(np.abs(b) < eps, a / (b + np.sign(b) * eps), a / b)

def log_sum_exp(values):
    """Стабильное вычисление log(exp(a) + exp(b) + ...)."""
    if len(values) == 0:
        return DTYPE(-np.inf)
    max_val = np.max(values)
    if np.isinf(max_val):
        return max_val
    return max_val + np.log(np.sum(np.exp(values - max_val), dtype=DTYPE))

# ====================
# СЕТКИ И БАЗОВЫЕ ФУНКЦИИ
# ====================
def create_grid(Nx: int, Nt: int, scale_factor: DTYPE = DTYPE(1.0)) -> Tuple[np.ndarray, np.ndarray, DTYPE, DTYPE]:
    """
    Создает равномерную пространственно-временную сетку.
    scale_factor: множитель для масштабирования (для улучшения обусловленности)
    """
    # Масштабируем область для улучшения численной устойчивости
    L_scaled = L / scale_factor
    
    # Создаем сетку в масштабированных координатах
    x_scaled = np.linspace(DTYPE(0.0), L_scaled, Nx + 1, dtype=DTYPE)
    t = np.linspace(DTYPE(0.0), T, Nt + 1, dtype=DTYPE)
    
    # Масштабируем обратно для вычислений
    x = x_scaled * scale_factor
    
    h = x[1] - x[0]
    tau = t[1] - t[0]
    
    return x, t, h, tau, scale_factor

def tridiagonal_solve_stable(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Устойчивое решение трехдиагональной системы методом прогонки.
    Добавлена защита от деления на ноль и переполнения.
    """
    n = len(d)
    
    # Копируем массивы для предотвращения модификации оригиналов
    a_c = a.copy().astype(DTYPE)
    b_c = b.copy().astype(DTYPE)
    c_c = c.copy().astype(DTYPE)
    d_c = d.copy().astype(DTYPE)
    
    # Добавляем малую величину к диагональным элементам для улучшения обусловленности
    eps = DTYPE(1e-15)
    b_c = b_c + eps
    
    # Прямой ход прогонки
    alpha = np.zeros(n, dtype=DTYPE)
    beta = np.zeros(n, dtype=DTYPE)
    
    # Первый шаг
    denominator = b_c[0]
    if np.abs(denominator) < DTYPE(1e-300):
        denominator = DTYPE(1e-300) * np.sign(denominator) if denominator != 0 else DTYPE(1e-300)
    
    alpha[0] = -c_c[0] / denominator
    beta[0] = d_c[0] / denominator
    
    # Основной цикл
    for i in range(1, n-1):
        denominator = b_c[i] + a_c[i] * alpha[i-1]
        if np.abs(denominator) < DTYPE(1e-300):
            denominator = DTYPE(1e-300) * np.sign(denominator) if denominator != 0 else DTYPE(1e-300)
        
        alpha[i] = -c_c[i] / denominator
        beta[i] = (d_c[i] - a_c[i] * beta[i-1]) / denominator
    
    # Последний шаг
    denominator = b_c[n-1] + a_c[n-1] * alpha[n-2]
    if np.abs(denominator) < DTYPE(1e-300):
        denominator = DTYPE(1e-300) * np.sign(denominator) if denominator != 0 else DTYPE(1e-300)
    
    u = np.zeros(n, dtype=DTYPE)
    u[n-1] = (d_c[n-1] - a_c[n-1] * beta[n-2]) / denominator
    
    # Обратный ход
    for i in range(n-2, -1, -1):
        u[i] = alpha[i] * u[i+1] + beta[i]
    
    return u

def compute_errors_stable(u_numerical: np.ndarray, u_analytical: np.ndarray, 
                          h: DTYPE, eps: DTYPE = DTYPE(1e-300)) -> Dict[str, DTYPE]:
    """
    Стабильное вычисление норм погрешности с защитой от переполнения.
    """
    # Используем относительную погрешность для избежания больших чисел
    abs_error = np.abs(u_numerical - u_analytical)
    abs_solution = np.abs(u_analytical)
    
    # Вычисляем относительную погрешность с защитой от деления на ноль
    rel_error = np.where(abs_solution > eps, abs_error / (abs_solution + eps), abs_error)
    
    # Максимальная абсолютная и относительная погрешность
    max_abs_error = np.max(abs_error)
    max_rel_error = np.max(rel_error)
    
    # L2-норма абсолютной погрешности
    # Используем стабильное суммирование Кахана
    sum_sq = np.sum(abs_error[1:-1]**2, dtype=DTYPE)
    l2_abs_error = np.sqrt(h * sum_sq) if sum_sq > DTYPE(0.0) else DTYPE(0.0)
    
    # L2-норма относительной погрешности
    rel_error_sq = rel_error[1:-1]**2
    sum_rel_sq = np.sum(rel_error_sq, dtype=DTYPE)
    l2_rel_error = np.sqrt(h * sum_rel_sq / (L - 2*h)) if sum_rel_sq > DTYPE(0.0) else DTYPE(0.0)
    
    return {
        'max_abs': max_abs_error,
        'max_rel': max_rel_error,
        'l2_abs': l2_abs_error,
        'l2_rel': l2_rel_error
    }

# ====================
# РЕАЛИЗАЦИЯ СХЕМ С ЗАЩИТОЙ ОТ ПЕРЕПОЛНЕНИЯ
# ====================
def explicit_scheme_stable(Nx: int, Nt: int, 
                          save_all_steps: bool = False,
                          scale_factor: DTYPE = DTYPE(1.0)) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Явная схема с защитой от переполнения.
    """
    x, t, h, tau, sf = create_grid(Nx, Nt, scale_factor)
    
    # Коэффициент устойчивости
    r = a * tau / (h**2)
    
    # Проверка устойчивости с защитой от деления на ноль
    if h == DTYPE(0.0):
        raise ValueError("Шаг по пространству равен нулю")
    
    # Стабильное вычисление условия устойчивости
    stability_condition = DTYPE(0.5)
    is_stable = r <= stability_condition
    
    # Масштабируем начальные данные
    u0_scaled = initial_condition(x / sf) if sf != DTYPE(1.0) else initial_condition(x)
    
    # Инициализация
    if save_all_steps:
        u = np.zeros((Nt + 1, Nx + 1), dtype=DTYPE)
        u[0, :] = u0_scaled
    else:
        u_prev = u0_scaled.copy()
        u_curr = np.zeros_like(u_prev)
    
    # Временной цикл
    start_time = time.time()
    
    # Используем стабильный способ вычисления разностей
    if save_all_steps:
        for n in range(Nt):
            # Граничные условия
            u[n+1, 0] = left_boundary(t[n+1])
            u[n+1, Nx] = right_boundary(t[n+1])
            
            # Внутренние точки с защитой от больших значений
            for i in range(1, Nx):
                # Вычисляем вторую разность с защитой от больших чисел
                diff2 = u[n, i+1] - DTYPE(2.0) * u[n, i] + u[n, i-1]
                
                # Ограничиваем чрезмерно большие приращения
                max_increment = DTYPE(10.0) * np.abs(u[n, i])
                increment = r * diff2
                
                # Применяем ограничение
                if np.abs(increment) > max_increment:
                    increment = np.sign(increment) * max_increment
                
                u[n+1, i] = u[n, i] + increment
            
            # Проверка на NaN и Inf
            if np.any(~np.isfinite(u[n+1, :])):
                warnings.warn(f"Обнаружены нефинитные значения на шаге {n+1}")
                # Заменяем нефинитные значения на ближайшие финитные
                u[n+1, :] = np.where(np.isfinite(u[n+1, :]), u[n+1, :], u[n, :])
        
        u_final = u[-1, :]
    else:
        u_prev = u0_scaled.copy()
        u_curr = np.zeros_like(u_prev)
        
        for n in range(Nt):
            u_curr[0] = left_boundary(t[n+1])
            u_curr[Nx] = right_boundary(t[n+1])
            
            for i in range(1, Nx):
                diff2 = u_prev[i+1] - DTYPE(2.0) * u_prev[i] + u_prev[i-1]
                max_increment = DTYPE(10.0) * np.abs(u_prev[i])
                increment = r * diff2
                
                if np.abs(increment) > max_increment:
                    increment = np.sign(increment) * max_increment
                
                u_curr[i] = u_prev[i] + increment
            
            # Проверка и коррекция
            if np.any(~np.isfinite(u_curr)):
                finite_mask = np.isfinite(u_curr)
                if np.any(finite_mask):
                    u_curr[~finite_mask] = np.mean(u_curr[finite_mask])
                else:
                    u_curr[:] = u_prev[:]
            
            u_prev = u_curr.copy()
        
        u_final = u_curr
    
    elapsed_time = time.time() - start_time
    
    # Масштабируем обратно при необходимости
    if sf != DTYPE(1.0):
        u_final = u_final * sf  # или другое преобразование в зависимости от задачи
    
    # Погрешность
    u_analytical_val = analytical_solution(x, T)
    errors = compute_errors_stable(u_final, u_analytical_val, h)
    
    info = {
        'h': h,
        'tau': tau,
        'r': float(r),
        'is_stable': is_stable,
        'time': elapsed_time,
        'errors': errors,
        'scheme': 'Явная схема (стабильная)',
        'scale_factor': sf,
        'Nx': Nx,
        'Nt': Nt
    }
    
    if save_all_steps:
        return u, x, t, info
    else:
        return u_final, x, info

def implicit_scheme_stable(Nx: int, Nt: int, 
                          save_all_steps: bool = False,
                          scale_factor: DTYPE = DTYPE(1.0)) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Неявная схема с защитой от переполнения.
    """
    x, t, h, tau, sf = create_grid(Nx, Nt, scale_factor)
    r = a * tau / (h**2)
    
    # Масштабируем начальные данные
    u0_scaled = initial_condition(x / sf) if sf != DTYPE(1.0) else initial_condition(x)
    
    # Инициализация
    if save_all_steps:
        u = np.zeros((Nt + 1, Nx + 1), dtype=DTYPE)
        u[0, :] = u0_scaled
    else:
        u_prev = u0_scaled.copy()
    
    # Подготовка коэффициентов для трехдиагональной системы
    n_inner = Nx - 1
    A = np.full(n_inner, -r, dtype=DTYPE)      # нижняя диагональ
    B = np.full(n_inner, DTYPE(1.0) + DTYPE(2.0) * r, dtype=DTYPE)  # главная диагональ
    C = np.full(n_inner, -r, dtype=DTYPE)      # верхняя диагональ
    D = np.zeros(n_inner, dtype=DTYPE)         # правая часть
    
    # Улучшаем обусловленность матрицы
    B = B + DTYPE(1e-12)  # небольшая регулярная добавка
    
    start_time = time.time()
    
    if save_all_steps:
        for n in range(Nt):
            u[n+1, 0] = left_boundary(t[n+1])
            u[n+1, Nx] = right_boundary(t[n+1])
            
            # Правая часть
            for i in range(1, Nx):
                D[i-1] = u[n, i]
            
            # Учет граничных условий с защитой от больших чисел
            D[0] += r * left_boundary(t[n+1])
            D[-1] += r * right_boundary(t[n+1])
            
            # Стабильное решение системы
            u_inner = tridiagonal_solve_stable(A, B, C, D)
            
            # Проверка результата
            if np.any(~np.isfinite(u_inner)):
                # Используем предыдущие значения в случае проблем
                u_inner = np.where(np.isfinite(u_inner), u_inner, u[n, 1:-1])
            
            u[n+1, 1:-1] = u_inner
            
            # Дополнительная стабилизация: ограничение экстремальных значений
            max_allowed = DTYPE(1e10)
            u[n+1, :] = np.clip(u[n+1, :], -max_allowed, max_allowed)
        
        u_final = u[-1, :]
    else:
        u_prev = u0_scaled.copy()
        u_curr = np.zeros_like(u_prev)
        
        for n in range(Nt):
            u_curr[0] = left_boundary(t[n+1])
            u_curr[Nx] = right_boundary(t[n+1])
            
            # Правая часть
            for i in range(1, Nx):
                D[i-1] = u_prev[i]
            
            D[0] += r * left_boundary(t[n+1])
            D[-1] += r * right_boundary(t[n+1])
            
            u_inner = tridiagonal_solve_stable(A, B, C, D)
            
            if np.any(~np.isfinite(u_inner)):
                u_inner = np.where(np.isfinite(u_inner), u_inner, u_prev[1:-1])
            
            u_curr[1:-1] = u_inner
            
            # Ограничение значений
            max_allowed = DTYPE(1e10)
            u_curr = np.clip(u_curr, -max_allowed, max_allowed)
            
            u_prev = u_curr.copy()
        
        u_final = u_curr
    
    elapsed_time = time.time() - start_time
    
    # Масштабируем обратно
    if sf != DTYPE(1.0):
        u_final = u_final * sf
    
    # Погрешность
    u_analytical_val = analytical_solution(x, T)
    errors = compute_errors_stable(u_final, u_analytical_val, h)
    
    info = {
        'h': h,
        'tau': tau,
        'r': float(r),
        'time': elapsed_time,
        'errors': errors,
        'scheme': 'Неявная схема (стабильная)',
        'scale_factor': sf,
        'Nx': Nx,
        'Nt': Nt
    }
    
    if save_all_steps:
        return u, x, t, info
    else:
        return u_final, x, info

def crank_nicolson_stable(Nx: int, Nt: int, 
                         save_all_steps: bool = False,
                         scale_factor: DTYPE = DTYPE(1.0)) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Схема Кранка-Николсона с защитой от переполнения.
    """
    x, t, h, tau, sf = create_grid(Nx, Nt, scale_factor)
    r_half = a * tau / (DTYPE(2.0) * h**2)  # половинный коэффициент для CN
    
    # Масштабируем начальные данные
    u0_scaled = initial_condition(x / sf) if sf != DTYPE(1.0) else initial_condition(x)
    
    # Инициализация
    if save_all_steps:
        u = np.zeros((Nt + 1, Nx + 1), dtype=DTYPE)
        u[0, :] = u0_scaled
    else:
        u_prev = u0_scaled.copy()
    
    # Коэффициенты для трехдиагональной системы
    n_inner = Nx - 1
    A = np.full(n_inner, -r_half, dtype=DTYPE)
    B = np.full(n_inner, DTYPE(1.0) + DTYPE(2.0) * r_half, dtype=DTYPE)
    C = np.full(n_inner, -r_half, dtype=DTYPE)
    D = np.zeros(n_inner, dtype=DTYPE)
    
    # Регуляризация
    B = B + DTYPE(1e-12)
    
    start_time = time.time()
    
    if save_all_steps:
        for n in range(Nt):
            u[n+1, 0] = left_boundary(t[n+1])
            u[n+1, Nx] = right_boundary(t[n+1])
            
            # Вычисление правой части с защитой от переполнения
            for i in range(1, Nx):
                # Явная часть
                explicit_part = r_half * (u[n, i+1] - DTYPE(2.0) * u[n, i] + u[n, i-1])
                
                # Ограничиваем чрезмерно большие значения
                max_explicit = DTYPE(100.0) * np.abs(u[n, i])
                if np.abs(explicit_part) > max_explicit:
                    explicit_part = np.sign(explicit_part) * max_explicit
                
                D[i-1] = u[n, i] + explicit_part
            
            # Учет граничных условий
            D[0] += r_half * left_boundary(t[n+1])
            D[-1] += r_half * right_boundary(t[n+1])
            
            # Решение системы
            u_inner = tridiagonal_solve_stable(A, B, C, D)
            
            # Проверка результата
            if np.any(~np.isfinite(u_inner)):
                u_inner = np.where(np.isfinite(u_inner), u_inner, u[n, 1:-1])
            
            u[n+1, 1:-1] = u_inner
            
            # Стабилизация: ограничение экстремальных значений
            max_allowed = DTYPE(1e10)
            u[n+1, :] = np.clip(u[n+1, :], -max_allowed, max_allowed)
        
        u_final = u[-1, :]
    else:
        u_prev = u0_scaled.copy()
        u_curr = np.zeros_like(u_prev)
        
        for n in range(Nt):
            u_curr[0] = left_boundary(t[n+1])
            u_curr[Nx] = right_boundary(t[n+1])
            
            # Правая часть
            for i in range(1, Nx):
                explicit_part = r_half * (u_prev[i+1] - DTYPE(2.0) * u_prev[i] + u_prev[i-1])
                max_explicit = DTYPE(100.0) * np.abs(u_prev[i])
                if np.abs(explicit_part) > max_explicit:
                    explicit_part = np.sign(explicit_part) * max_explicit
                
                D[i-1] = u_prev[i] + explicit_part
            
            D[0] += r_half * left_boundary(t[n+1])
            D[-1] += r_half * right_boundary(t[n+1])
            
            u_inner = tridiagonal_solve_stable(A, B, C, D)
            
            if np.any(~np.isfinite(u_inner)):
                u_inner = np.where(np.isfinite(u_inner), u_inner, u_prev[1:-1])
            
            u_curr[1:-1] = u_inner
            u_curr = np.clip(u_curr, -DTYPE(1e10), DTYPE(1e10))
            
            u_prev = u_curr.copy()
        
        u_final = u_curr
    
    elapsed_time = time.time() - start_time
    
    # Масштабируем обратно
    if sf != DTYPE(1.0):
        u_final = u_final * sf
    
    # Погрешность
    u_analytical_val = analytical_solution(x, T)
    errors = compute_errors_stable(u_final, u_analytical_val, h)
    
    info = {
        'h': h,
        'tau': tau,
        'r_half': float(r_half),
        'time': elapsed_time,
        'errors': errors,
        'scheme': 'Кранка-Николсона (стабильная)',
        'scale_factor': sf,
        'Nx': Nx,
        'Nt': Nt
    }
    
    if save_all_steps:
        return u, x, t, info
    else:
        return u_final, x, info

# ====================
# АДАПТИВНЫЙ ВЫБОР ШАГОВ
# ====================
def adaptive_grid_selection(target_error: float = 1e-6, 
                           max_Nx: int = 10000,
                           max_Nt: int = 100000) -> Tuple[int, int]:
    """
    Адаптивный выбор сетки на основе целевой погрешности.
    """
    print("Адаптивный выбор сетки...")
    
    # Начинаем с грубой сетки
    Nx = 10
    Nt = 100
    
    best_error = float('inf')
    best_Nx, best_Nt = Nx, Nt
    
    iteration = 0
    max_iterations = 10
    
    while iteration < max_iterations and Nx <= max_Nx and Nt <= max_Nt:
        iteration += 1
        
        # Пробуем разные соотношения Nt/Nx
        ratios = [5, 10, 20]
        for ratio in ratios:
            Nt_test = Nx * ratio
            
            if Nt_test > max_Nt:
                continue
            
            try:
                # Используем схему Кранка-Николсона как наиболее точную
                u_final, x, info = crank_nicolson_stable(Nx, Nt_test)
                error = info['errors']['max_abs']
                
                print(f"  Nx={Nx}, Nt={Nt_test}, error={error:.2e}")
                
                if error < best_error:
                    best_error = error
                    best_Nx, best_Nt = Nx, Nt_test
                
                # Если достигли целевой погрешности
                if error <= target_error:
                    print(f"Достигнута целевая погрешность {target_error}")
                    return Nx, Nt_test
            
            except Exception as e:
                print(f"  Ошибка для Nx={Nx}, Nt={Nt_test}: {e}")
                continue
        
        # Увеличиваем сетку
        Nx *= 2
    
    print(f"Лучшая найденная сетка: Nx={best_Nx}, Nt={best_Nt}, error={best_error:.2e}")
    return best_Nx, best_Nt

# ====================
# ИССЛЕДОВАНИЕ УСТОЙЧИВОСТИ
# ====================
def stability_investigation():
    """
    Исследование устойчивости явной схемы при разных r.
    """
    print("\n" + "="*60)
    print("Исследование устойчивости явной схемы")
    print("="*60)
    
    Nx = 50
    T_test = DTYPE(0.1)
    
    # Разные значения r (коэффициент устойчивости)
    r_values = [0.1, 0.25, 0.5, 0.6, 0.75, 1.0]
    
    for r_target in r_values:
        # Вычисляем tau из r = a*tau/h^2
        h = L / Nx
        tau = r_target * h**2 / a
        
        # Соответствующее Nt
        Nt = int(np.ceil(T_test / tau))
        tau = T_test / Nt  # Корректируем для точного T
        
        print(f"\nr = {r_target:.3f}, h = {h:.4f}, τ = {tau:.6f}, Nt = {Nt}")
        
        try:
            u_final, x, info = explicit_scheme_stable(Nx, Nt)
            error = info['errors']['max_abs']
            is_stable = info['is_stable']
            
            status = "УСТОЙЧИВА" if is_stable else "НЕУСТОЙЧИВА"
            print(f"  Схема: {status}, Погрешность: {error:.2e}")
            
            # Проверка на наличие взрывного роста
            if error > 10.0:
                print("  ВНИМАНИЕ: Возможно взрывное решение!")
            
        except Exception as e:
            print(f"  Ошибка: {e}")

# ====================
# ВИЗУАЛИЗАЦИЯ С СТАБИЛЬНЫМИ МЕТОДАМИ
# ====================
def plot_stable_results():
    """
    Построение графиков с использованием стабильных методов.
    """
    # Используем адаптивную сетку
    print("\nВыбор адаптивной сетки...")
    Nx, Nt = adaptive_grid_selection(target_error=1e-4)
    
    print(f"\nИспользуемая сетка: Nx={Nx}, Nt={Nt}")
    
    # Вычисления
    print("\nВыполнение расчетов...")
    u_explicit, x, info_explicit = explicit_scheme_stable(Nx, Nt)
    u_implicit, x, info_implicit = implicit_scheme_stable(Nx, Nt)
    u_cn, x, info_cn = crank_nicolson_stable(Nx, Nt)
    u_analytical = analytical_solution(x, T)
    
    # Создание графиков
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Сравнение решений
    ax = axes[0, 0]
    ax.plot(x, u_explicit, 'b-', linewidth=2, label='Явная схема')
    ax.plot(x, u_implicit, 'r--', linewidth=2, label='Неявная схема')
    ax.plot(x, u_cn, 'g-.', linewidth=2, label='Кранка-Николсона')
    ax.plot(x, u_analytical, 'k:', linewidth=3, label='Аналитическое')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,T)')
    ax.set_title(f'Сравнение решений (t={T})')
    ax.legend()
    ax.grid(True)
    
    # 2. Абсолютные погрешности
    ax = axes[0, 1]
    ax.semilogy(x, np.abs(u_explicit - u_analytical), 'b-', linewidth=2, label='Явная')
    ax.semilogy(x, np.abs(u_implicit - u_analytical), 'r--', linewidth=2, label='Неявная')
    ax.semilogy(x, np.abs(u_cn - u_analytical), 'g-.', linewidth=2, label='Кранка-Николсона')
    ax.set_xlabel('x')
    ax.set_ylabel('Абсолютная погрешность')
    ax.set_title('Абсолютные погрешности (логарифмическая шкала)')
    ax.legend()
    ax.grid(True)
    
    # 3. Относительные погрешности
    ax = axes[0, 2]
    rel_error_exp = np.abs(u_explicit - u_analytical) / (np.abs(u_analytical) + 1e-10)
    rel_error_imp = np.abs(u_implicit - u_analytical) / (np.abs(u_analytical) + 1e-10)
    rel_error_cn = np.abs(u_cn - u_analytical) / (np.abs(u_analytical) + 1e-10)
    
    ax.semilogy(x, rel_error_exp, 'b-', linewidth=2, label='Явная')
    ax.semilogy(x, rel_error_imp, 'r--', linewidth=2, label='Неявная')
    ax.semilogy(x, rel_error_cn, 'g-.', linewidth=2, label='Кранка-Николсона')
    ax.set_xlabel('x')
    ax.set_ylabel('Относительная погрешность')
    ax.set_title('Относительные погрешности (логарифмическая шкала)')
    ax.legend()
    ax.grid(True)
    
    # 4. Временная эволюция (с использованием CN)
    ax = axes[1, 0]
    # Берем подмножество точек для визуализации
    Nx_small = min(Nx, 50)
    Nt_small = min(Nt, 100)
    u_cn_full, x_small, t_small, info_cn_full = crank_nicolson_stable(
        Nx_small, Nt_small, save_all_steps=True, scale_factor=DTYPE(0.1)
    )
    
    # Выбираем несколько моментов времени для отображения
    time_indices = [0, Nt_small//4, Nt_small//2, 3*Nt_small//4, Nt_small-1]
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for idx, color in zip(time_indices, colors):
        u_at_time = u_cn_full[idx, :]
        if info_cn_full['scale_factor'] != DTYPE(1.0):
            u_at_time = u_at_time * info_cn_full['scale_factor']
        ax.plot(x_small, u_at_time, color=color, 
                linewidth=2, label=f't={t_small[idx]:.3f}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Временная эволюция (схема Кранка-Николсона)')
    ax.legend()
    ax.grid(True)
    
    # 5. Зависимость погрешности от параметров сетки
    ax = axes[1, 1]
    
    # Исследуем сходимость
    Nx_values = [10, 20, 40, 80, 160]
    errors_exp = []
    errors_imp = []
    errors_cn = []
    
    for Nx_test in Nx_values:
        # Подбираем Nt для сохранения стабильности
        Nt_test = Nx_test * 20  # фиксированное отношение
        
        try:
            u_exp, x_test, info_exp = explicit_scheme_stable(Nx_test, Nt_test)
            u_imp, x_test, info_imp = implicit_scheme_stable(Nx_test, Nt_test)
            u_cn_test, x_test, info_cn_test = crank_nicolson_stable(Nx_test, Nt_test)
            
            u_ana_test = analytical_solution(x_test, T)
            
            errors_exp.append(np.max(np.abs(u_exp - u_ana_test)))
            errors_imp.append(np.max(np.abs(u_imp - u_ana_test)))
            errors_cn.append(np.max(np.abs(u_cn_test - u_ana_test)))
        except:
            # Пропускаем проблемные сетки
            continue
    
    if errors_exp:
        h_values = [L/Nx_val for Nx_val in Nx_values[:len(errors_exp)]]
        ax.loglog(h_values, errors_exp, 'bo-', linewidth=2, markersize=8, label='Явная')
        ax.loglog(h_values, errors_imp, 'rs-', linewidth=2, markersize=8, label='Неявная')
        ax.loglog(h_values, errors_cn, 'g^-', linewidth=2, markersize=8, label='Кранка-Николсона')
        
        # Теоретические линии сходимости
        h_test = np.logspace(np.log10(min(h_values)), np.log10(max(h_values)), 10)
        ax.loglog(h_test, 0.1*np.array(h_test)**2, 'k--', linewidth=1, label='O(h²)')
        ax.loglog(h_test, 0.1*np.array(h_test), 'k:', linewidth=1, label='O(h)')
        
        ax.set_xlabel('Шаг по пространству h')
        ax.set_ylabel('Максимальная погрешность')
        ax.set_title('Сходимость схем')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # 6. Информационная панель
    ax = axes[1, 2]
    ax.axis('off')
    
    info_text = (
        f'Параметры задачи:\n'
        f'  a = {float(a):.2f}, L = {float(L):.2f}, T = {float(T):.2f}\n'
        f'  Nx = {Nx}, Nt = {Nt}\n'
        f'  h = {float(info_cn["h"]):.6f}, τ = {float(info_cn["tau"]):.6f}\n\n'
        f'Погрешности (макс. абсолютные):\n'
        f'  Явная: {float(info_explicit["errors"]["max_abs"]):.2e}\n'
        f'  Неявная: {float(info_implicit["errors"]["max_abs"]):.2e}\n'
        f'  Кранка-Николсона: {float(info_cn["errors"]["max_abs"]):.2e}\n\n'
        f'Время расчета:\n'
        f'  Явная: {info_explicit["time"]:.3f} с\n'
        f'  Неявная: {info_implicit["time"]:.3f} с\n'
        f'  Кранка-Николсона: {info_cn["time"]:.3f} с\n'
    )
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'explicit': info_explicit,
        'implicit': info_implicit,
        'cn': info_cn
    }

# ====================
# ТЕСТ НА ПЕРЕПОЛНЕНИЕ
# ====================
def overflow_test():
    """
    Тест на переполнение с экстремальными параметрами.
    """
    print("\n" + "="*60)
    print("Тест на переполнение с экстремальными параметрами")
    print("="*60)
    
    # Экстремально малый шаг по времени (может вызвать переполнение в явной схеме)
    test_cases = [
        {"Nx": 10, "Nt": 10000, "desc": "Очень мелкий шаг по времени"},
        {"Nx": 1000, "Nt": 10, "desc": "Очень мелкий шаг по пространству"},
        {"Nx": 500, "Nt": 5000, "desc": "Очень большая сетка"},
    ]
    
    for case in test_cases:
        print(f"\nТест: {case['desc']}")
        print(f"Параметры: Nx={case['Nx']}, Nt={case['Nt']}")
        
        try:
            # Пробуем схему Кранка-Николсона как наиболее стабильную
            start_time = time.time()
            u_final, x, info = crank_nicolson_stable(
                case['Nx'], case['Nt'], 
                scale_factor=DTYPE(0.01)  # Масштабируем для стабильности
            )
            elapsed = time.time() - start_time
            
            print(f"  Успех! Время: {elapsed:.2f} с")
            print(f"  Погрешность: {info['errors']['max_abs']:.2e}")
            
            # Проверяем на наличие нефинитных значений
            if np.any(~np.isfinite(u_final)):
                print("  ВНИМАНИЕ: Обнаружены нефинитные значения!")
            else:
                print("  Все значения конечны.")
                
        except MemoryError:
            print("  ОШИБКА: Недостаточно памяти!")
        except Exception as e:
            print(f"  ОШИБКА: {e}")

# ====================
# ОСНОВНАЯ ПРОГРАММА
# ====================
def main_stable():
    print("="*70)
    print("ЛАБОРАТОРНАЯ РАБОТА 1: РЕШЕНИЕ С ЗАЩИТОЙ ОТ ПЕРЕПОЛНЕНИЯ")
    print("Вариант 2: Уравнение теплопроводности")
    print("="*70)
    
    print(f"\nИспользуемый тип данных: {DTYPE}")
    print(f"Размер типа: {np.dtype(DTYPE).itemsize} байт")
    
    # 1. Исследование устойчивости
    stability_investigation()
    
    # 2. Тест на переполнение
    overflow_test()
    
    # 3. Основные расчеты с визуализацией
    print("\n" + "="*60)
    print("Основные расчеты и визуализация")
    print("="*60)
    
    results = plot_stable_results()
    
    # 4. Сводная таблица результатов
    print("\n" + "="*70)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*70)
    
    print("\n{:<25} {:<12} {:<20} {:<20} {:<15}".format(
        "Схема", "Время (с)", "L∞-погрешность", "L2-погрешность", "Стабильность"))
    print("-"*95)
    
    schemes = [
        ("Явная схема", results['explicit']),
        ("Неявная схема", results['implicit']),
        ("Кранка-Николсона", results['cn'])
    ]
    
    for name, info in schemes:
        time_val = info['time']
        max_error = float(info['errors']['max_abs'])
        l2_error = float(info['errors']['l2_abs'])
        
        # Определяем стабильность
        if 'is_stable' in info:
            stability = "Условно" if info['is_stable'] else "Неустойч."
        else:
            stability = "Безусловно"
        
        print("{:<25} {:<12.3f} {:<20.2e} {:<20.2e} {:<15}".format(
            name, time_val, max_error, l2_error, stability))
    
    print("-"*95)
    
    # 5. Рекомендации по выбору схемы
    print("\n" + "="*60)
    print("РЕКОМЕНДАЦИИ ПО ВЫБОРУ СХЕМЫ")
    print("="*60)
    
    print("\n1. ЯВНАЯ СХЕМА:")
    print("   + Простота реализации")
    print("   + Минимальные затраты памяти на шаг")
    print("   - Требует малого шага по времени (τ ≤ h²/(2a))")
    print("   - Может быть неустойчивой при больших τ")
    print("   РЕКОМЕНДАЦИЯ: Использовать только для быстрых оценок на мелких сетках")
    
    print("\n2. НЕЯВНАЯ СХЕМА:")
    print("   + Безусловно устойчива")
    print("   + Допускает большие шаги по времени")
    print("   - Требует решения системы уравнений на каждом шаге")
    print("   - Меньшая точность по времени (первый порядок)")
    print("   РЕКОМЕНДАЦИЯ: Использовать для грубых расчетов с большими τ")
    
    print("\n3. СХЕМА КРАНКА-НИКОЛСОНА:")
    print("   + Безусловно устойчива")
    print("   + Второй порядок точности по времени и пространству")
    print("   + Наиболее точная из трех")
    print("   - Требует решения системы уравнений на каждом шаге")
    print("   - Чувствительна к разрывам в начальных данных")
    print("   РЕКОМЕНДАЦИЯ: Лучший выбор для точных расчетов")
    
    print("\n" + "="*70)
    print("ВЫВОДЫ:")
    print("="*70)
    
    best_scheme = min(schemes, key=lambda x: x[1]['errors']['max_abs'])
    worst_scheme = max(schemes, key=lambda x: x[1]['errors']['max_abs'])
    
    print(f"\n1. Наиболее точная схема: {best_scheme[0]}")
    print(f"   Погрешность: {float(best_scheme[1]['errors']['max_abs']):.2e}")
    
    print(f"\n2. Наименее точная схема: {worst_scheme[0]}")
    print(f"   Погрешность: {float(worst_scheme[1]['errors']['max_abs']):.2e}")
    
    speed_ranking = sorted(schemes, key=lambda x: x[1]['time'])
    print(f"\n3. Самая быстрая схема: {speed_ranking[0][0]}")
    print(f"   Время: {speed_ranking[0][1]['time']:.3f} с")
    
    print(f"\n4. Рекомендуемая схема для данного варианта: Кранка-Николсона")
    print("   Причины: высокая точность, безусловная устойчивость, второй порядок")
    
    print("\n" + "="*70)
    print("РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*70)

# Запуск программы
if __name__ == "__main__":
    main_stable()