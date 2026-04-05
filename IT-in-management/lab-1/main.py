import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

installs = pd.read_csv('Installs.csv')
installs.columns = ['Cohort', 'Installs']

revenue_raw = pd.read_csv('Revenue Cohort.csv', header=None)

revenue = revenue_raw.iloc[2:].copy()
revenue.drop(columns=[1], inplace=True) 

week_cols = [str(int(x)) for x in revenue_raw.iloc[0, 2:].values if pd.notna(x)]
revenue = revenue.iloc[:, :len(week_cols)+1]
revenue.columns = ['Cohort'] + week_cols

revenue['Cohort'] = revenue['Cohort'].astype(int)

for c in week_cols:
    revenue[c] = revenue[c].astype(str).str.replace('$', '', regex=False)\
                                         .str.replace('\xa0', '', regex=False)\
                                         .str.replace(' ', '', regex=False)\
                                         .str.replace(',', '.', regex=False)
    revenue[c] = pd.to_numeric(revenue[c], errors='coerce')

df = pd.merge(revenue, installs, on='Cohort')

cols_3m = [str(i) for i in range(13)]
df_3m = df.dropna(subset=cols_3m) 

total_installs_3m = df_3m['Installs'].sum()
total_rev_3m = df_3m[cols_3m].sum().sum()

arpu_3m = total_rev_3m / total_installs_3m

cumulative_arpu = []
weeks = []

for w in range(len(week_cols)):
    col = str(w)
    valid_cohorts = df.dropna(subset=[col])
    if len(valid_cohorts) > 0:
        rev_sum = valid_cohorts[col].sum()
        inst_sum = valid_cohorts['Installs'].sum()
        arpu_w = rev_sum / inst_sum
        
        if w == 0:
            cumulative_arpu.append(arpu_w)
        else:
            cumulative_arpu.append(cumulative_arpu[-1] + arpu_w)
        weeks.append(w)

x = np.array(weeks[1:]) 
y = np.array(cumulative_arpu[1:])
x_log = np.log(x).reshape(-1, 1)

model = LinearRegression()
model.fit(x_log, y)
y_pred = model.predict(x_log)

a = model.coef_[0]
b = model.intercept_
r2 = r2_score(y, y_pred)

arpu_1y = model.predict(np.log([[51]]))[0]

plt.figure(figsize=(10, 6))
plt.scatter(weeks, cumulative_arpu, color='blue', label='Фактичні дані (Cumulative ARPU)')

x_plot = np.arange(1, 55)
y_plot = model.predict(np.log(x_plot.reshape(-1, 1)))
plt.plot(x_plot, y_plot, color='red', linestyle='--', label='Логарифмічна регресія (Прогноз)')

plt.axvline(x=12, color='green', linestyle=':', label='3 місяці (12 тижнів)')
plt.axvline(x=51, color='orange', linestyle=':', label='1 рік (51 тиждень)')
plt.scatter([51], [arpu_1y], color='orange', s=100, zorder=5)

formula = f"Формула: y = {a:.4f} * ln(x) + {b:.4f}\n$R^2$ = {r2:.4f}"
plt.text(20, 0.20, formula, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.title('Прогноз Cumulative ARPU (LTV) за допомогою лінійної регресії (ln(x))')
plt.xlabel('Тижні з моменту встановлення')
plt.ylabel('Cumulative ARPU, $')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('arpu_forecast.png')

print(f"ARPU 3 місяці: ${arpu_3m:.4f}")
print(f"Прогноз ARPU 1 рік: ${arpu_1y:.4f}")