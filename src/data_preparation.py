import pandas as pd

print("Generating feature reports...")

train = pd.read_csv("../data/processed/train.csv")
val = pd.read_csv("../data/processed/val.csv")

df = pd.concat([train, val], ignore_index=True)

human = df[df['label'] == 1]
ai = df[df['label'] == 0]

# 1. Elongations
elong_report = pd.DataFrame({
    'class': ['Human', 'AI'],
    'count': [human['elongations'].sum(), ai['elongations'].sum()]
})
elong_report.to_csv("../reports/feature_elongations_report.csv", index=False, encoding="utf-8-sig")

# 2. Periods
periods_report = pd.DataFrame({
    'class': ['Human', 'AI'],
    'count': [human['periods'].sum(), ai['periods'].sum()]
})
periods_report.to_csv("../reports/feature_periods_report.csv", index=False, encoding="utf-8-sig")

# 3. Verbs
verbs_report = pd.DataFrame({
    'class': ['Human', 'AI'],
    'count': [human['verbs'].sum(), ai['verbs'].sum()]
})
verbs_report.to_csv("../reports/feature_verbs_report.csv", index=False, encoding="utf-8-sig")

# 4. Duals
duals_report = pd.DataFrame({
    'class': ['Human', 'AI'],
    'count': [human['duals'].sum(), ai['duals'].sum()]
})
duals_report.to_csv("../reports/feature_duals_report.csv", index=False, encoding="utf-8-sig")

# 5. Entity Diversity (ratio)
diversity_report = pd.DataFrame({
    'class': ['Human', 'AI'],
    'diversity_ratio': [human['entity_diversity'].mean(), ai['entity_diversity'].mean()]
})
diversity_report.to_csv("../reports/feature_diversity_report.csv", index=False, encoding="utf-8-sig")

print("\n reports saved")
