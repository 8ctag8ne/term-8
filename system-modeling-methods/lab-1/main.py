import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# --- 1. Параметри та генерація ---
lambda_val = 2.0  # Ваш обраний параметр
n = 3000          # Кількість чисел за умовою
np.random.seed(42)

# xi - рівномірно розподілена на (0, 1)
xi = np.random.uniform(0.0001, 1, n) 
# x_i за формулою 9-го варіанту
x = lambda_val / xi

# --- 2. Обчислення характеристик (емпірично) ---
mean_emp = np.mean(x)
var_emp = np.var(x, ddof=1)

print(f"Емпіричне середнє: {mean_emp:.4f}")
print(f"Емпірична дисперсія: {var_emp:.4f}")

# --- 3. Побудова гістограми та графіка щільності ---
plt.figure(figsize=(10, 6))
# Обмежуємо діапазон для візуалізації, бо у розподілу Парето дуже довгий "хвіст"
plot_limit = lambda_val * 20
plt.hist(x, bins=50, range=(lambda_val, plot_limit), density=True, 
         alpha=0.6, color='skyblue', edgecolor='black', label='Гістограма частот')

# Теоретична щільність f(x) = lambda / x^2
x_theo = np.linspace(lambda_val, plot_limit, 1000)
y_theo = lambda_val / (x_theo**2)
plt.plot(x_theo, y_theo, 'r-', lw=2, label=f'Теоретична щільність ($f(x) = {lambda_val}/x^2$)')

plt.title(f'Перевірка закону розподілу (Варіант 9, n={n}, $\lambda={lambda_val}$)')
plt.xlabel('Значення x')
plt.ylabel('Щільність')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("pareto_distribution.png")

# --- 4. Перевірка за критерієм хі-квадрат (Пірсона) ---
k = 10  # Кількість інтервалів
# Використовуємо квантилі для рівноймовірних інтервалів (краще для "важких хвостів")
probs = np.linspace(0, 1, k + 1)
# Квантилі для Парето: x = lambda / (1 - p)
bin_edges = lambda_val / (1 - probs[:-1])
bin_edges = np.append(bin_edges, [np.inf])

observed, _ = np.histogram(x, bins=bin_edges)
expected = np.full(k, n / k)  # Очікувана кількість у кожному інтервалі

chi_sq_stat = np.sum((observed - expected)**2 / expected)
df = k - 1  # Ступені свободи
p_value = 1 - chi2.cdf(chi_sq_stat, df)

print("\n--- Результати критерію χ² ---")
print(f"Статистика хі-квадрат: {chi_sq_stat:.4f}")
print(f"Ступені свободи: {df}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value > alpha:
    print(f"Висновок: Оскільки p-value > {alpha}, гіпотеза про відповідність закону розподілу ПІДТВЕРДЖУЄТЬСЯ.")
else:
    print(f"Висновок: Оскільки p-value < {alpha}, гіпотеза про відповідність закону розподілу ВІДХИЛЯЄТЬСЯ.")