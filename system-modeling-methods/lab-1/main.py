import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate

# --- Допоміжні функції ---

def manual_stats(data, n):
    """Розрахунок середнього та дисперсії вручну"""
    s = 0
    for val in data:
        s += val
    mean_val = s / n
    
    v = 0
    for val in data:
        v += (val - mean_val)**2
    variance_val = v / (n - 1)
    
    return mean_val, variance_val

def calculate_pareto_params_mle(data):
    """
    Оцінка параметрів розподілу Парето методом максимальної правдоподібності (MLE).
    """
    n = len(data)
    # Оцінка xm (scale) - це мінімальне значення вибірки
    xm_est = np.min(data)
    
    # Оцінка alpha (shape) - через логарифмічне рівняння правдоподібності
    sum_ln = np.sum(np.log(data / xm_est))
    alpha_est = n / sum_ln
    
    return xm_est, alpha_est

def pareto_cdf(val, xm, alpha):
    """Функція розподілу F(x) = 1 - (xm/x)^alpha"""
    if val < xm:
        return 0
    return 1 - (xm / val)**alpha

def print_chi_square_table(final_bins, final_ni, n, xm_param, alpha_param):
    """Формування таблиці частот та розрахунок статистики хі-квадрат"""
    table_data = []
    theoretical_counts = []
    chi_sq_total = 0
    
    for i in range(len(final_ni)):
        a, b = final_bins[i], final_bins[i+1]
        obs = final_ni[i]
        
        # Теоретична ймовірність попадання в інтервал
        p_i = pareto_cdf(b, xm_param, alpha_param) - pareto_cdf(a, xm_param, alpha_param)
        
        # Корекція для останнього інтервалу (замикання хвоста)
        if i == len(final_ni) - 1: 
             p_i = 1 - pareto_cdf(a, xm_param, alpha_param)

        expected_n = n * p_i
        theoretical_counts.append(expected_n)
        
        # Розрахунок доданку (O - E)^2 / E
        term = ((obs - expected_n)**2) / expected_n if expected_n > 0 else 0
        chi_sq_total += term
        
        interval_str = f"[{a:7.4f}, {b:7.4f})"
        table_data.append([i + 1, interval_str, obs, f"{expected_n:.2f}", f"{term:.4f}"])
    
    # Підсумковий рядок
    table_data.append(["", "РАЗОМ:", sum(final_ni), f"{sum(theoretical_counts):.2f}", f"{chi_sq_total:.4f}"])
    
    print("\nСтатистична таблиця перевірки гіпотези:")
    headers = ["№", "Інтервал [a, b)", "n_i", "np_i^T", "Внесок (chi-sq)"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="center"))
    
    return chi_sq_total

# --- Основний блок ---

# 1. Генерація даних
n = 3000
lambd_gen = 0.01

seed = int(time.time())
np.random.seed(seed)
xi = np.random.uniform(0, 1, n)
# Захист від ділення на нуль
xi = np.maximum(xi, 1e-10) 
x = lambd_gen / xi

# 2. Розрахунок емпіричних характеристик (для звіту)
mean_emp, var_emp = manual_stats(x, n)

print(f"Емпіричне середнє: {mean_emp:.6f}")
print(f"Емпірична дисперсія: {var_emp:.6f}")

# 3. Визначення параметрів розподілу (Метод ММП)
xm_calc, alpha_calc = calculate_pareto_params_mle(x)

print("-" * 50)
print(f"Оцінка параметрів (MLE): x_m = {xm_calc:.6f}, alpha = {alpha_calc:.6f}")
print("-" * 50)

# 4. Побудова інтервального ряду
x_sorted = sorted(x)
limit_95 = x_sorted[int(n * 0.95)] 
min_x = min(x)

# Розбиття на інтервали
k_initial = 50 
h = (limit_95 - min_x) / k_initial
bins = [min_x + i * h for i in range(k_initial)]
bins.append(max(x)) 

# Підрахунок частот
frequencies = [0] * (len(bins) - 1)
for val in x:
    for i in range(len(bins) - 1):
        if bins[i] <= val < bins[i+1]:
            frequencies[i] += 1
            break
    if val == bins[-1]: 
        frequencies[-1] += 1

# Об'єднання малочисельних інтервалів
final_bins = [bins[0]]
final_ni = []
current_n = 0

for i in range(len(frequencies)):
    current_n += frequencies[i]
    if current_n >= 5 or i == len(frequencies) - 1:
        final_bins.append(bins[i+1])
        final_ni.append(current_n)
        current_n = 0

if final_ni[-1] < 5 and len(final_ni) > 1:
    last_n = final_ni.pop()
    final_ni[-1] += last_n
    final_bins.pop(-2)

# 5. Перевірка за критерієм Хі-квадрат
chi_sq_val = print_chi_square_table(final_bins, final_ni, n, xm_calc, alpha_calc)

# 6. Порівняння з критичним значенням
k_final = len(final_ni)
q = 2 # Кількість оцінених параметрів (xm, alpha)
df = k_final - 1 - q
alpha_level = 0.05

if df <= 0:
    print("Помилка: недостатньо інтервалів для перевірки.")
else:
    chi_crit = stats.chi2.ppf(1 - alpha_level, df)
    print("-" * 65)
    print(f"Ступені вільності (k-1-2): {df}")
    print(f"Хі-квадрат розрах.: {chi_sq_val:.4f}")
    print(f"Хі-квадрат критич.: {chi_crit:.4f}")

    if chi_sq_val < chi_crit:
        print("РЕЗУЛЬТАТ: Гіпотеза ПРИЙНЯТА")
    else:
        print("РЕЗУЛЬТАТ: Гіпотеза ВІДХИЛЕНА")

# 7. Візуалізація
plt.figure(figsize=(10, 6))

# Гістограма
plt.hist(x[x < limit_95], bins=30, density=False, alpha=0.6, color='skyblue', label='Емпіричні частоти')

# Теоретична крива MLE
line_x = np.linspace(xm_calc, limit_95, 1000)   
line_y = (alpha_calc * (xm_calc**alpha_calc)) / (line_x**(alpha_calc + 1)) * n * (bins[1] - bins[0]) # Масштабування для порівняння з гістограмою

plt.plot(line_x, line_y, 'r-', lw=2, label=f'Апроксимація (ММП)')
plt.title(f'Перевірка розподілу Парето')
plt.xlabel('X')
plt.ylabel('Частота')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('lab1_result.png')