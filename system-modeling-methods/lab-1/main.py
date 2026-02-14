import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Параметри та генерація
n = 3000
lambd = 0.01  # Обраний параметр за умовою
alpha_theory = 1.0  # Теоретичний показник форми для вашого генератора

# Генеруємо xi ~ U(0, 1) та перетворюємо за формулою x = lambda / xi
seed = int(time.time())
np.random.seed(seed)
xi = np.random.uniform(0, 1, n)
x = lambd / xi

# 2. Ручний розрахунок емпіричних характеристик
def manual_stats(data):
    # Середнє значення
    s = 0
    for val in data:
        s += val
    mean_val = s / n
    
    # Дисперсія (незміщена)
    v = 0
    for val in data:
        v += (val - mean_val)**2
    variance_val = v / (n - 1)
    
    return mean_val, variance_val

mean_emp, var_emp = manual_stats(x)

print(f"Емпіричне середнє: {mean_emp:.6f}")
print(f"Емпірична дисперсія: {var_emp:.6f}")
print("-" * 50)

# 3. Побудова інтервального ряду та перевірка хі-квадрат вручну
# Для розподілу Парето з довгим хвостом використовуємо 95-й перцентиль як межу для інтервалів
x_sorted = sorted(x)
limit_95 = x_sorted[int(n * 0.95)] # Відсікаємо 5% екстремальних значень для стабільності інтервалів
min_x = min(x)

# Початкове розбиття (35 інтервалів до 95-го перцентиля + 1 для "хвоста")
k_initial = 35
h = (limit_95 - min_x) / k_initial
bins = [min_x + i * h for i in range(k_initial)]
bins.append(max(x)) # Останній інтервал закриває весь "хвіст" до нескінченності

# Підрахунок частот n_i
frequencies = [0] * (len(bins) - 1)
for val in x:
    for i in range(len(bins) - 1):
        if bins[i] <= val < bins[i+1]:
            frequencies[i] += 1
            break
    if val == bins[-1]: # Крайнє значення
        frequencies[-1] += 1

# Об'єднання інтервалів, де n_i < 5 (умова застосування хі-квадрат)
final_bins = [bins[0]]
final_ni = []
current_n = 0

for i in range(len(frequencies)):
    current_n += frequencies[i]
    if current_n >= 5 or i == len(frequencies) - 1:
        final_bins.append(bins[i+1])
        final_ni.append(current_n)
        current_n = 0

# Якщо останній інтервал після циклу все ще < 5, об'єднуємо його з передостаннім
if final_ni[-1] < 5 and len(final_ni) > 1:
    last_n = final_ni.pop()
    final_ni[-1] += last_n
    final_bins.pop(-2)

# 4. Розрахунок теоретичних частот np_i та критерію
# Для Парето F(x) = 1 - (lambda/x)^alpha. Тут alpha = 1.
def pareto_cdf(val, l, a):
    return 1 - (l / val)**a

chi_sq_val = 0
theoretical_counts = []

print(f"{'Інтервал [a, b)':<25} | {'n_i':<5} | {'np_i^T':<8} | {'Доданок chi^2'}")
print("-" * 65)

for i in range(len(final_ni)):
    a, b = final_bins[i], final_bins[i+1]
    # Теоретична ймовірність p_i = F(b) - F(a)
    p_i = pareto_cdf(b, lambd, alpha_theory) - pareto_cdf(a, lambd, alpha_theory)
    expected_n = n * p_i
    theoretical_counts.append(expected_n)
    
    # Доданок (n_i - np_i)^2 / np_i
    term = ((final_ni[i] - expected_n)**2) / expected_n
    chi_sq_val += term
    print(f"[{a:7.4f}, {b:7.4f}) | {final_ni[i]:5} | {expected_n:8.2f} | {term:.4f}")

# 5. Перевірка результату
k_final = len(final_ni)
# Ступені вільності: k - 1 - q. Ми знаємо параметри генератора, тому q = 0.
# Якщо ви оцінювали lambda як min(x), то q = 1.
df = k_final - 1 
alpha_level = 0.05
chi_crit = stats.chi2.ppf(1 - alpha_level, df)

print("-" * 65)
print(f"Сумарне значення Chi-square: {chi_sq_val:.4f}")
print(f"Критичне значення (df={df}): {chi_crit:.4f}")

if chi_sq_val < chi_crit:
    print("РЕЗУЛЬТАТ: Гіпотеза ПРИЙНЯТА (Відповідає розподілу Парето)")
else:
    print("РЕЗУЛЬТАТ: Гіпотеза ВІДХИЛЕНА")

# 6. Візуалізація
plt.figure(figsize=(10, 6))

# Гістограма для основної маси даних (до 95 перцентиля)
plt.hist(x[x < limit_95], bins=30, density=True, alpha=0.6, color='skyblue', label='Емпірична щільність (95%)')

# Теоретична крива щільності f(x) = alpha * lambda^alpha / x^(alpha + 1)
# При alpha = 1: f(x) = lambda / x^2
line_x = np.linspace(min_x, limit_95, 500)
line_y = lambd / (line_x**2)
plt.plot(line_x, line_y, 'r-', lw=2, label='Теоретична щільність Парето')

plt.title(f'Перевірка розподілу (Варіант 9, $\lambda={lambd}$)\n$\chi^2={chi_sq_val:.2f} < \chi^2_{{kp}}={chi_crit:.2f}$')
plt.xlabel('Значення X')
plt.ylabel('Щільність')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('lab1_result.png')