"""
Лабораторна робота: Апроксимація функціональної залежності
Метод найменших квадратів (МНК)

Варіант функції: y = b₀ + b₁·x₁² + b₂·√((x₂+2)³) + b₃·x₃

Послідовність виконання:
1. Перевірка на мультиколінеарність (алгоритм Фаррара-Глобера)
2. МНК-ідентифікація параметрів
3. Таблиця результатів
4. Критерій найменших квадратів Φ
5. Графік порівняння
6. Кореляційно-регресійний аналіз
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

# Налаштування виведення
np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.precision', 4)
pd.set_option('display.width', 120)

# Створення директорії для результатів
os.makedirs("output", exist_ok=True)

print("=" * 100)
print("ЛАБОРАТОРНА РОБОТА: АПРОКСИМАЦІЯ ФУНКЦІОНАЛЬНОЇ ЗАЛЕЖНОСТІ")
print("Метод найменших квадратів (МНК)")
print("=" * 100)

# ============================================================================
# ВХІДНІ ДАНІ
# ============================================================================
print("\n" + "=" * 100)
print("ВХІДНІ ДАНІ")
print("=" * 100)

data = np.array([
    [3,  4,  6,  1],
    [7,  10, 13, 96],
    [9,  12, 15, 155],
    [12, 8,  13, 296],
    [10, 14, 14, 179],
    [13, 15, 18, 322],
    [20, 22, 19, 780],
    [24, 25, 21, 1075],
    [26, 26, 28, 1288],
    [28, 28, 30, 1494],
    [26, 27, 28, 1280],
    [28, 31, 29, 1466],
    [27, 30, 34, 1379],
    [26, 33, 38, 1259],
    [29, 35, 40, 1700],
    [31, 36, 41, 1811],
    [46, 46, 35, 3950],
    [48, 49, 37, 4300],
    [49, 50, 38, 4542],
    [53, 51, 42, 5500],
], dtype=float)

x1 = data[:, 0]  # Перша вхідна змінна
x2 = data[:, 1]  # Друга вхідна змінна
x3 = data[:, 2]  # Третя вхідна змінна
y  = data[:, 3]  # Вихідна змінна (відгук)
n  = len(y)      # Кількість спостережень

print(f"\nКількість спостережень: n = {n}")
print(f"Кількість вхідних змінних: m = 3")
print(f"\nВигляд апроксимуючої функції:")
print("y = b₀ + b₁·x₁² + b₂·√((x₂+2)³) + b₃·x₃")

print("\nПерші 5 спостережень:")
df_initial = pd.DataFrame({
    '№': range(1, 6),
    'x₁': x1[:5],
    'x₂': x2[:5],
    'x₃': x3[:5],
    'y': y[:5]
})
print(df_initial.to_string(index=False))

# ============================================================================
# ЗАВДАННЯ 1. ПЕРЕВІРКА НА МУЛЬТИКОЛІНЕАРНІСТЬ
# Алгоритм Фаррара-Глобера
# ============================================================================
print("\n\n" + "=" * 100)
print("ЗАВДАННЯ 1. ПЕРЕВІРКА ВХІДНИХ ЗМІННИХ НА МУЛЬТИКОЛІНЕАРНІСТЬ")
print("Алгоритм Фаррара-Глобера")
print("=" * 100)

# Матриця вхідних змінних
X_input = np.column_stack([x1, x2, x3])
m = X_input.shape[1]  # Кількість змінних

# --- Крок 1. Стандартизація ---
print("\n--- Крок 1. Стандартизація вхідних змінних ---")
print("Формула: x*ᵢₖ = (xᵢₖ - x̄ₖ) / sₖ")

X_mean = np.mean(X_input, axis=0)
X_std = np.std(X_input, axis=0, ddof=1)

print(f"\nСередні значення:")
print(f"  x̄₁ = {X_mean[0]:.4f}")
print(f"  x̄₂ = {X_mean[1]:.4f}")
print(f"  x̄₃ = {X_mean[2]:.4f}")

print(f"\nСтандартні відхилення:")
print(f"  s₁ = {X_std[0]:.4f}")
print(f"  s₂ = {X_std[1]:.4f}")
print(f"  s₃ = {X_std[2]:.4f}")

# Стандартизація
X_standardized = (X_input - X_mean) / X_std

# --- Крок 2. Кореляційна матриця ---
print("\n--- Крок 2. Кореляційна матриця ---")
print("Формула: R = (1/(n-1)) Z^T Z")

R_corr = (X_standardized.T @ X_standardized) / (n - 1)

print("\nКореляційна матриця R:")
df_corr = pd.DataFrame(R_corr,
                       columns=['x₁', 'x₂', 'x₃'],
                       index=['x₁', 'x₂', 'x₃'])
print(df_corr.round(4))

# --- Крок 3. χ²-тест загальної мультиколінеарності ---
print("\n--- Крок 3. χ²-тест загальної мультиколінеарності ---")
print("Формула: χ² = -(n - 1 - (2m+5)/6) · ln(det R)")

det_R = np.linalg.det(R_corr)
chi2_calc = -(n - 1 - (2*m + 5)/6) * np.log(det_R)
df_chi2 = m * (m - 1) // 2

alpha = 0.05
chi2_crit = stats.chi2.ppf(1 - alpha, df_chi2)

print(f"\ndet(R) = {det_R:.6f}")
print(f"χ²розр = {chi2_calc:.4f}")
print(f"Ступені свободи: ν = m(m-1)/2 = {df_chi2}")
print(f"χ²крит(α={alpha}, ν={df_chi2}) = {chi2_crit:.4f}")

if chi2_calc > chi2_crit:
    print(f"\nВИСНОВОК: χ²розр ({chi2_calc:.4f}) > χ²крит ({chi2_crit:.4f})")
    print("=> Серед змінних x₁, x₂, x₃ ІСНУЄ мультиколінеарність")
else:
    print(f"\nВИСНОВОК: χ²розр ({chi2_calc:.4f}) ≤ χ²крит ({chi2_crit:.4f})")
    print("=> Мультиколінеарність відсутня")

# --- Крок 4. F-тест для кожного регресора ---
print("\n--- Крок 4. F-тест для кожного регресора ---")
print("Перевірка мультиколінеарності кожної змінної з рештою")
print("Формула: Fᵢ = (cᵢᵢ - 1) · (n - m) / (m - 1)")

# Обернення матриці
R_inv = np.linalg.inv(R_corr)

F_crit = stats.f.ppf(1 - alpha, m - 1, n - m)

print(f"\nКритичне значення: Fкрит(α={alpha}, ν₁={m-1}, ν₂={n-m}) = {F_crit:.4f}")
print("\nРезультати F-тесту:")

for i in range(m):
    C_ii = R_inv[i, i]
    R2_i = 1 - 1/C_ii
    F_i = (C_ii - 1) * (n - m) / (m - 1)
    
    status = "МУЛЬТИКОЛІНЕАРНА" if F_i > F_crit else "не мультиколінеарна"
    print(f"  x{i+1}: cᵢᵢ={C_ii:.4f}, R²ᵢ={R2_i:.4f}, Fᵢ={F_i:.4f} => {status}")

# --- Крок 5. t-тест для парних кореляцій ---
print("\n--- Крок 5. t-тест для парних кореляцій ---")
print("Формула: t(xᵢ,xⱼ) = r(xᵢ,xⱼ) · √((n-m)/(1-r²(xᵢ,xⱼ)))")

t_crit = stats.t.ppf(1 - alpha/2, n - m)
print(f"\nКритичне значення: tкрит(α/2={alpha/2}, ν={n-m}) = {t_crit:.4f}")
print("\nРезультати t-тесту для пар:")

for i in range(m):
    for j in range(i+1, m):
        # Частинний коефіцієнт кореляції
        r_partial = -R_inv[i, j] / np.sqrt(R_inv[i, i] * R_inv[j, j])
        
        # t-статистика для парної кореляції
        r_pair = R_corr[i, j]
        t_calc = r_pair * np.sqrt(n - m) / np.sqrt(max(1 - r_pair**2, 1e-12))
        
        status = "МУЛЬТИКОЛІНЕАРНІ" if abs(t_calc) > t_crit else "не мультиколінеарні"
        print(f"  (x{i+1}, x{j+1}): r={r_pair:.4f}, r*={r_partial:.4f}, t={t_calc:.4f} => {status}")

print("\n" + "-" * 100)
print("ЗАГАЛЬНИЙ ВИСНОВОК ПО МУЛЬТИКОЛІНЕАРНОСТІ:")
print("Виявлено мультиколінеарність між деякими змінними.")
print("Проте продовжуємо побудову моделі для демонстрації методу МНК.")
print("-" * 100)

# ============================================================================
# ЗАВДАННЯ 2. МНК-ІДЕНТИФІКАЦІЯ ПАРАМЕТРІВ
# ============================================================================
print("\n\n" + "=" * 100)
print("ЗАВДАННЯ 2. ІДЕНТИФІКАЦІЯ ПАРАМЕТРІВ ФУНКЦІЇ МЕТОДОМ МНК")
print("=" * 100)

print("\nФункція: y = b₀ + b₁·x₁² + b₂·√((x₂+2)³) + b₃·x₃")

# --- Крок 2.1: Формування базисних функцій ---
print("\n--- Крок 2.1. Формування базисних функцій ---")

phi0 = np.ones(n)
phi1 = x1**2
phi2 = np.sqrt((x2 + 2)**3)
phi3 = x3

print("Базисні функції:")
print("  φ₀(x) = 1")
print("  φ₁(x) = x₁²")
print("  φ₂(x) = √((x₂+2)³)")
print("  φ₃(x) = x₃")

# --- Крок 2.2: Формування матриці планування X ---
print("\n--- Крок 2.2. Формування матриці планування X ---")

X = np.column_stack([phi0, phi1, phi2, phi3])
k = X.shape[1] - 1  # Кількість регресорів (без вільного члена)

print(f"\nРозмірність матриці X: {X.shape[0]} × {X.shape[1]}")
print("\nПерші 5 рядків матриці X:")
df_X = pd.DataFrame(X[:5], 
                    columns=['φ₀=1', 'φ₁=x₁²', 'φ₂=√((x₂+2)³)', 'φ₃=x₃'])
print(df_X.round(4).to_string(index=False))

# --- Крок 2.3: Обчислення X^T X ---
print("\n--- Крок 2.3. Обчислення X^T X ---")
XtX = X.T @ X

print(f"\nРозмірність X^T X: {XtX.shape[0]} × {XtX.shape[1]}")
print("\nМатриця X^T X:")
df_XtX = pd.DataFrame(XtX,
                      columns=['b₀', 'b₁', 'b₂', 'b₃'],
                      index=['b₀', 'b₁', 'b₂', 'b₃'])
print(df_XtX.round(2))

# --- Крок 2.4: Обчислення X^T y ---
print("\n--- Крок 2.4. Обчислення X^T y ---")
Xty = X.T @ y

print("\nВектор X^T y:")
for i, val in enumerate(Xty):
    print(f"  (X^T y)[{i}] = {val:.4f}")

# --- Крок 2.5: Знаходження (X^T X)^(-1) ---
print("\n--- Крок 2.5. Знаходження оберненої матриці (X^T X)^(-1) ---")
print("Метод: використання numpy.linalg.inv()")

XtX_inv = np.linalg.inv(XtX)

print("\nМатриця (X^T X)^(-1):")
df_XtX_inv = pd.DataFrame(XtX_inv,
                          columns=['b₀', 'b₁', 'b₂', 'b₃'],
                          index=['b₀', 'b₁', 'b₂', 'b₃'])
print(df_XtX_inv)

# Перевірка
check = XtX @ XtX_inv
print("\nПеревірка: X^T X · (X^T X)^(-1) ≈ I:")
print(np.round(check, 6))

# --- Крок 2.6: Обчислення вектора параметрів b ---
print("\n--- Крок 2.6. Обчислення вектора параметрів b ---")
print("Формула: b = (X^T X)^(-1) X^T y")

b = XtX_inv @ Xty

print("\nОтримані параметри функції:")
print(f"  b₀ = {b[0]:.6f}")
print(f"  b₁ = {b[1]:.6f}")
print(f"  b₂ = {b[2]:.6f}")
print(f"  b₃ = {b[3]:.6f}")

print("\n" + "-" * 100)
print("АПРОКСИМУЮЧА ФУНКЦІЯ:")
print(f"y = {b[0]:.6f} + {b[1]:.6f}·x₁² + {b[2]:.6f}·√((x₂+2)³) + {b[3]:.6f}·x₃")
print("-" * 100)

# ============================================================================
# ЗАВДАННЯ 3. ТАБЛИЦЯ РЕЗУЛЬТАТІВ
# ============================================================================
print("\n\n" + "=" * 100)
print("ЗАВДАННЯ 3. ТАБЛИЦЯ ЗАДАНИХ ТА РОЗРАХОВАНИХ ЗНАЧЕНЬ")
print("=" * 100)

# Обчислення розрахованих значень
y_hat = X @ b

# Залишки (відхилення)
residuals = y - y_hat

print("\nПовна таблиця результатів:")
df_results = pd.DataFrame({
    '№': range(1, n+1),
    'x₁': x1,
    'x₂': x2,
    'x₃': x3,
    'y (спостереж.)': y,
    'ŷ (розрахов.)': y_hat,
    'e = y - ŷ': residuals
})
print(df_results.round(4).to_string(index=False))

# ============================================================================
# ЗАВДАННЯ 4. КРИТЕРІЙ НАЙМЕНШИХ КВАДРАТІВ
# ============================================================================
print("\n\n" + "=" * 100)
print("ЗАВДАННЯ 4. ОБЧИСЛЕННЯ КРИТЕРІЮ НАЙМЕНШИХ КВАДРАТІВ")
print("=" * 100)

print("\nКритерій МНК:")
print("Φ(b) = Σ(yᵢ - ŷᵢ)² = ||y - Xb||²")

Phi = np.sum(residuals**2)

print(f"\nΦ = {Phi:.4f}")

# Додаткова статистика
print(f"\nМаксимальне відхилення: {np.max(np.abs(residuals)):.4f}")
print(f"Середнє відхилення: {np.mean(residuals):.4f}")
print(f"Середнє абсолютне відхилення: {np.mean(np.abs(residuals)):.4f}")

# ============================================================================
# ЗАВДАННЯ 5. ГРАФІК ПОРІВНЯННЯ
# ============================================================================
print("\n\n" + "=" * 100)
print("ЗАВДАННЯ 5. ГРАФІЧНЕ ПОРІВНЯННЯ РЕЗУЛЬТАТІВ")
print("=" * 100)

# Графік 1: Порівняння реальних та розрахованих значень
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(1, n+1), y, 'bo-', label='Спостережувані y', linewidth=2, markersize=8)
ax.plot(range(1, n+1), y_hat, 'r^--', label='Розраховані ŷ', linewidth=2, markersize=8)
ax.set_xlabel("Номер спостереження", fontsize=12)
ax.set_ylabel("Значення y", fontsize=12)
ax.set_title("Порівняння спостережуваних та розрахованих значень", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/1_comparison_plot.png", dpi=300)
plt.close()
print("\n✓ Графік порівняння збережено: output/1_comparison_plot.png")

# Графік 2: Залишки
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if r >= 0 else 'red' for r in residuals]
ax.bar(range(1, n+1), residuals, color=colors, edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', linewidth=2, linestyle='-')
ax.set_xlabel("Номер спостереження", fontsize=12)
ax.set_ylabel("Залишок (y - ŷ)", fontsize=12)
ax.set_title("Залишки регресійної моделі", fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("output/2_residuals_plot.png", dpi=300)
plt.close()
print("✓ Графік залишків збережено: output/2_residuals_plot.png")

# Графік 3: Діаграма розсіювання (фактичні vs розраховані)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y, y_hat, s=100, alpha=0.6, edgecolors='black')
# Лінія ідеального збігу
min_val = min(y.min(), y_hat.min())
max_val = max(y.max(), y_hat.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ідеальний збіг')
ax.set_xlabel("Спостережувані значення y", fontsize=12)
ax.set_ylabel("Розраховані значення ŷ", fontsize=12)
ax.set_title("Діаграма розсіювання: y vs ŷ", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/3_scatter_plot.png", dpi=300)
plt.close()
print("✓ Діаграму розсіювання збережено: output/3_scatter_plot.png")

# ============================================================================
# ЗАВДАННЯ 6. КОРЕЛЯЦІЙНО-РЕГРЕСІЙНИЙ АНАЛІЗ
# ============================================================================
print("\n\n" + "=" * 100)
print("ЗАВДАННЯ 6. КОРЕЛЯЦІЙНО-РЕГРЕСІЙНИЙ АНАЛІЗ")
print("=" * 100)

# --- 6.1. Коефіцієнт детермінації R² ---
print("\n--- 6.1. Коефіцієнт детермінації R² ---")
print("Формула: R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²")

y_mean = np.mean(y)
SS_res = np.sum((y - y_hat)**2)  # Залишкова сума квадратів
SS_tot = np.sum((y - y_mean)**2)  # Загальна сума квадратів
R2 = 1 - SS_res / SS_tot

print(f"\nСереднє значення: ȳ = {y_mean:.4f}")
print(f"Залишкова сума квадратів: SS_res = {SS_res:.4f}")
print(f"Загальна сума квадратів: SS_tot = {SS_tot:.4f}")
print(f"\nКоефіцієнт детермінації: R² = {R2:.6f}")
print(f"Це означає, що {R2*100:.2f}% варіації y пояснюється моделлю")

# --- 6.2. Коефіцієнт кореляції r ---
print("\n--- 6.2. Коефіцієнт кореляції r ---")
print("Формула: r = √R²")

r = np.sqrt(R2)
print(f"\nКоефіцієнт кореляції: r = {r:.6f}")

# --- 6.3. Шкала Чеддока ---
print("\n--- 6.3. Оцінка щільності зв'язку за шкалою Чеддока ---")

chaddock_scale = [
    (0.1, 0.3, "слабкий зв'язок"),
    (0.3, 0.5, "помірний зв'язок"),
    (0.5, 0.7, "помітний зв'язок"),
    (0.7, 0.9, "сильний зв'язок"),
    (0.9, 0.99, "дуже сильний зв'язок"),
    (0.99, 1.0, "практично функціональний зв'язок")
]

quality = "невизначено"
for low, high, desc in chaddock_scale:
    if low <= r < high:
        quality = desc
        break

print(f"\nШкала Чеддока:")
print(f"  r = {r:.4f} => {quality.upper()}")

# --- 6.4. Дисперсія залишків ---
print("\n--- 6.4. Дисперсія залишків ---")
print("Формула: σ² = Φ / (n - k - 1)")

sigma2 = Phi / (n - k - 1)
sigma = np.sqrt(sigma2)

print(f"\nДисперсія залишків: σ² = {sigma2:.4f}")
print(f"Стандартне відхилення: σ = {sigma:.4f}")

# --- 6.5. Коваріаційна матриця параметрів ---
print("\n--- 6.5. Коваріаційна матриця та стандартні похибки параметрів ---")
print("Формула: Cov(b) = σ² · (X^T X)^(-1)")

cov_b = sigma2 * XtX_inv
se_b = np.sqrt(np.diag(cov_b))

print("\nКоваріаційна матриця Cov(b):")
df_cov = pd.DataFrame(cov_b,
                      columns=['b₀', 'b₁', 'b₂', 'b₃'],
                      index=['b₀', 'b₁', 'b₂', 'b₃'])
print(df_cov)

print("\nСтандартні похибки параметрів:")
for i in range(len(b)):
    print(f"  SE(b{i}) = {se_b[i]:.6f}")

# --- 6.6. Перевірка адекватності моделі (F-критерій) ---
print("\n--- 6.6. Перевірка адекватності моделі (F-критерій) ---")
print("Формула: F = (R² / k) / ((1 - R²) / (n - k - 1))")

F_calc = (R2 / k) / ((1 - R2) / (n - k - 1))
F_crit_model = stats.f.ppf(1 - alpha, k, n - k - 1)

print(f"\nF-статистика: F = {F_calc:.4f}")
print(f"Критичне значення: F_крит(α={alpha}, ν₁={k}, ν₂={n-k-1}) = {F_crit_model:.4f}")

if F_calc > F_crit_model:
    print(f"\nВИСНОВОК: F ({F_calc:.4f}) > F_крит ({F_crit_model:.4f})")
    print("=> Модель є АДЕКВАТНОЮ, коефіцієнт кореляції статистично значимий")
else:
    print(f"\nВИСНОВОК: F ({F_calc:.4f}) ≤ F_крит ({F_crit_model:.4f})")
    print("=> Модель неадекватна")

# --- 6.7. Статистична значимість коефіцієнтів (t-критерій) ---
print("\n--- 6.7. Статистична значимість коефіцієнтів (t-критерій) ---")
print("Формула: t(bᵢ) = bᵢ / SE(bᵢ)")

t_stats = b / se_b
t_crit_coef = stats.t.ppf(1 - alpha/2, n - k - 1)

print(f"\nКритичне значення: t_крит(α/2={alpha/2}, ν={n-k-1}) = {t_crit_coef:.4f}")
print("\nРезультати t-тесту для параметрів:")

for i in range(len(b)):
    significant = "ЗНАЧИМИЙ" if abs(t_stats[i]) > t_crit_coef else "незначимий"
    print(f"  b{i}: b={b[i]:10.6f}, SE={se_b[i]:10.6f}, t={t_stats[i]:8.4f} => {significant}")

# --- 6.8. Довірчі інтервали параметрів ---
print("\n--- 6.8. Довірчі інтервали параметрів ---")
print("Формула: bᵢ ∈ [bᵢ - t_крит·SE(bᵢ); bᵢ + t_крит·SE(bᵢ)]")

print(f"\nДовірчі інтервали (рівень довіри {1-alpha}):")
for i in range(len(b)):
    lower = b[i] - t_crit_coef * se_b[i]
    upper = b[i] + t_crit_coef * se_b[i]
    contains_zero = "так" if lower <= 0 <= upper else "НІ"
    print(f"  b{i}: [{lower:10.6f}; {upper:10.6f}]  (містить 0: {contains_zero})")

# --- 6.9. Факторна та загальна дисперсії ---
print("\n--- 6.9. Факторна та загальна дисперсії ---")

sigma_fact2 = np.sum((y_hat - y_mean)**2) / (n - 1)
sigma_total2 = np.sum((y - y_mean)**2) / (n - 1)

print(f"\nФакторна дисперсія: σ²_факт = {sigma_fact2:.4f}")
print(f"Загальна дисперсія: σ²_заг = {sigma_total2:.4f}")
print(f"Відношення: σ²_факт / σ²_заг = {sigma_fact2/sigma_total2:.6f} = R²")

# ============================================================================
# ПІДСУМКОВІ РЕЗУЛЬТАТИ
# ============================================================================
print("\n\n" + "=" * 100)
print("ПІДСУМКОВІ РЕЗУЛЬТАТИ АНАЛІЗУ")
print("=" * 100)

print("\n1. ОТРИМАНА МОДЕЛЬ:")
print(f"   y = {b[0]:.6f} + {b[1]:.6f}·x₁² + {b[2]:.6f}·√((x₂+2)³) + {b[3]:.6f}·x₃")

print("\n2. ЯКІСТЬ МОДЕЛІ:")
print(f"   • Критерій МНК: Φ = {Phi:.4f}")
print(f"   • Коефіцієнт детермінації: R² = {R2:.6f} ({R2*100:.2f}%)")
print(f"   • Коефіцієнт кореляції: r = {r:.6f}")
print(f"   • Оцінка за Чеддоком: {quality}")
print(f"   • Дисперсія залишків: σ² = {sigma2:.4f}")

print("\n3. СТАТИСТИЧНА ЗНАЧИМІСТЬ:")
print(f"   • F-критерій адекватності: F = {F_calc:.4f} (F_крит = {F_crit_model:.4f})")
print(f"   • Модель: {'АДЕКВАТНА' if F_calc > F_crit_model else 'НЕАДЕКВАТНА'}")
print(f"   • Значимі параметри: ", end="")
significant_params = [f"b{i}" for i in range(len(b)) if abs(t_stats[i]) > t_crit_coef]
print(", ".join(significant_params) if significant_params else "немає")

print("\n4. МУЛЬТИКОЛІНЕАРНІСТЬ:")
print(f"   • χ²-тест: χ² = {chi2_calc:.4f} (χ²_крит = {chi2_crit:.4f})")
print(f"   • Висновок: {'Виявлено мультиколінеарність' if chi2_calc > chi2_crit else 'Відсутня'}")

print("\n5. ГРАФІЧНІ РЕЗУЛЬТАТИ:")
print("   • output/1_comparison_plot.png - порівняння y та ŷ")
print("   • output/2_residuals_plot.png - залишки моделі")
print("   • output/3_scatter_plot.png - діаграма розсіювання")

print("\n" + "=" * 100)
print("АНАЛІЗ ЗАВЕРШЕНО")
print("=" * 100)