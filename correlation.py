import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("metrics/corr_mat.csv", index_col=0)

df_cleaned = df.fillna(0).round(3)
df_cleaned.to_csv('./corr_mat.csv')

plt.figure(figsize=(20, 10))
sns.heatmap(df_cleaned, annot=False, cmap='coolwarm', center=0)

plt.title("Correlation Matrix Heatmap", fontsize=18)
plt.tight_layout()
plt.show()


threshold = 0.7

correlated_pairs = []
cols = df_cleaned.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        corr_val = df_cleaned.iloc[i, j]
        if abs(corr_val) >= threshold:
            correlated_pairs.append((cols[i], cols[j], corr_val))

correlated_pairs = sorted(correlated_pairs, key=lambda x: abs(x[2]), reverse=True)

print("Coppie di feature con correlazione elevata:\n")
for f1, f2, val in correlated_pairs:
    print(f"{f1} <--> {f2} | Correlation: {val:.3f}")