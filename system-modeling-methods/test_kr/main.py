import numpy as np

# Параметри моделі (Варіант 9)
lam = 0.5
t1_avg = 1.8
l1_limit = 5

mu2 = 2.0
t2_avg = 1.0 / mu2

p_leave = 0.25 # 1/4 ймовірність виходу
t3_avg = 4.0

T_max = 10000

# Ініціалізація змінних стану
t_curr = 0
t_next_in = np.random.exponential(1/lam)
t_out1 = t_out2 = t_out3 = float('inf')

q1 = q2 = q3 = 0

n_all = n_un = n_serv_sys = 0
n_serv1 = n_serv2 = n_serv3 = 0 # Для розрахунку часу очікування
sum_l1 = sum_l2 = sum_l3 = 0

while t_curr < T_max:
    # Визначення найближчої події
    t_min = min(t_next_in, t_out1, t_out2, t_out3)
    dt = t_min - t_curr

    # Збір статистики (накопичення довжини черг)
    sum_l1 += q1 * dt
    sum_l2 += q2 * dt
    sum_l3 += q3 * dt

    t_curr = t_min

    # Подія 0: Надходження вимоги
    if t_min == t_next_in:
        n_all += 1
        if q1 < l1_limit:
            if t_out1 == float('inf'):
                t_out1 = t_curr + np.random.exponential(t1_avg)
            else:
                q1 += 1
        else:
            n_un += 1 # Відмова через переповнення l1
        t_next_in = t_curr + np.random.exponential(1/lam)

    # Подія 1: Завершення в K1
    elif t_min == t_out1:
        n_serv1 += 1
        if t_out2 == float('inf'):
            t_out2 = t_curr + np.random.exponential(t2_avg)
        else:
            q2 += 1 # Очікує перед K2

        if q1 > 0:
            q1 -= 1
            t_out1 = t_curr + np.random.exponential(t1_avg)
        else:
            t_out1 = float('inf')

    # Подія 2: Завершення в K2
    elif t_min == t_out2:
        n_serv2 += 1
        # Розгалуження: 1/4 - вихід, 3/4 - на K3
        if np.random.rand() > p_leave:
            if t_out3 == float('inf'):
                t_out3 = t_curr + np.random.exponential(t3_avg)
            else:
                q3 += 1 # Очікує перед K3
        else:
            n_serv_sys += 1 # Успішно покинула систему

        if q2 > 0:
            q2 -= 1
            t_out2 = t_curr + np.random.exponential(t2_avg)
        else:
            t_out2 = float('inf')

    # Подія 3: Завершення в K3
    elif t_min == t_out3:
        n_serv3 += 1
        n_serv_sys += 1
        if q3 > 0:
            q3 -= 1
            t_out3 = t_curr + np.random.exponential(t3_avg)
        else:
            t_out3 = float('inf')

# Розрахунок показників
p_rej = n_un / n_all if n_all > 0 else 0
l1_aver = sum_l1 / T_max
l2_aver = sum_l2 / T_max
l3_aver = sum_l3 / T_max

# Середній час очікування = (інтеграл черги) / (кількість обслугованих у вузлі)
w1_aver = sum_l1 / n_serv1 if n_serv1 > 0 else 0
w2_aver = sum_l2 / n_serv2 if n_serv2 > 0 else 0
w3_aver = sum_l3 / n_serv3 if n_serv3 > 0 else 0

print(f"Ймовірність відмови (p_rej) = {p_rej:.4f}")
print(f"Середня довжина черги 1 (l1_aver) = {l1_aver:.4f}")
print(f"Середня довжина черги 2 (l2_aver) = {l2_aver:.4f}")
print(f"Середня довжина черги 3 (l3_aver) = {l3_aver:.4f}")
print(f"Середній час очікування в черзі 1 (w1_aver) = {w1_aver:.4f}")
print(f"Середній час очікування в черзі 2 (w2_aver) = {w2_aver:.4f}")
print(f"Середній час очікування в черзі 3 (w3_aver) = {w3_aver:.4f}")