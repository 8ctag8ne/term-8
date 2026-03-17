import math
from typing import Dict

def calculate_closed_mmo_detailed() -> Dict:
    N = 25
    r1, r2, r3 = 2, 1, 2  
    mu1, mu2, mu3 = 25/4.0, 20/4.0, 60/8.0  
    
    # 1. Розрахунок коефіцієнтів передачі (e_i)
    e1 = 1.0
    e2 = 0.4 * e1
    e3 = 0.3 * e1

    # 2. Визначення допоміжних функцій p_i(k)
    def calc_p_i(k: int, e_i: float, mu_i: float, r_i: int) -> float:
        rho = e_i / mu_i
        if k <= r_i:
            return (rho ** k) / math.factorial(k)
        else:
            return (rho ** k) / (math.factorial(r_i) * (r_i ** (k - r_i)))

    p1_vals = [calc_p_i(k, e1, mu1, r1) for k in range(N + 1)]
    p2_vals = [calc_p_i(k, e2, mu2, r2) for k in range(N + 1)]
    p3_vals = [calc_p_i(k, e3, mu3, r3) for k in range(N + 1)]

    # 3. Розрахунок нормуючого множника C(N)
    sum_for_C = 0
    for k1 in range(N + 1):
        for k2 in range(N + 1 - k1):
            k3 = N - k1 - k2
            sum_for_C += p1_vals[k1] * p2_vals[k2] * p3_vals[k3]
    
    C_N = 1.0 / sum_for_C

    # 4. Визначення функцій P_CMO_i(j)
    P_CMO1 = [0.0] * (N + 1)
    P_CMO2 = [0.0] * (N + 1)
    P_CMO3 = [0.0] * (N + 1)

    for k1 in range(N + 1):
        for k2 in range(N + 1 - k1):
            k3 = N - k1 - k2
            prob = C_N * p1_vals[k1] * p2_vals[k2] * p3_vals[k3]
            P_CMO1[k1] += prob
            P_CMO2[k2] += prob
            P_CMO3[k3] += prob

    # 5. Розрахунок показників ефективності
    def calc_metrics(P_CMO: list, r_i: int, mu_i: float) -> dict:
        L_i = sum((j - r_i) * P_CMO[j] for j in range(r_i + 1, N + 1))
        R_i = r_i - sum((r_i - j) * P_CMO[j] for j in range(r_i))
        M_i = L_i + R_i
        lambda_i = R_i * mu_i
        T_i = M_i / lambda_i if lambda_i > 0 else 0
        Q_i = L_i / lambda_i if lambda_i > 0 else 0
        return {"L": L_i, "R": R_i, "M": M_i, "lambda": lambda_i, "T": T_i, "Q": Q_i}

    return {
        "e": [e1, e2, e3],
        "C_N": C_N,
        "p_vals": [p1_vals, p2_vals, p3_vals],
        "P_CMO": [P_CMO1, P_CMO2, P_CMO3],
        "metrics": [calc_metrics(P_CMO1, r1, mu1), calc_metrics(P_CMO2, r2, mu2), calc_metrics(P_CMO3, r3, mu3)]
    }

res = calculate_closed_mmo_detailed()

print(f"Коефіцієнти передачі: e1={res['e'][0]}, e2={res['e'][1]}, e3={res['e'][2]}")
print(f"Нормуючий множник C(N) = {res['C_N']:.6e}\n")

for i in range(3):
    print(f"--- СМО {i+1} ---")
    print(f"Допоміжні функції p_{i+1}(k) для k=0,1,2: {[round(x, 4) for x in res['p_vals'][i][:3]]} ...")
    print(f"Ймовірності станів P_{i+1}(j) для j=0,1,2: {[round(x, 4) for x in res['P_CMO'][i][:3]]} ...")
    print(f"Показники: L={res['metrics'][i]['L']:.4f}, R={res['metrics'][i]['R']:.4f}, M={res['metrics'][i]['M']:.4f}")
    print(f"Час: T={res['metrics'][i]['T']:.4f} год, Q={res['metrics'][i]['Q']:.4f} год\n")