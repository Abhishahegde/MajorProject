import pandas as pd

# =====================
# Dataset 1 preprocessing
# =====================
df1 = pd.read_excel("external_triggers_lifestyle.xlsx")

# Drop rows with missing Severity
df1 = df1.dropna(subset=["Severity"])

# Round Severity to nearest integer for classification
df1["Severity"] = df1["Severity"].round().astype(int)

# Ensure categorical columns are strings
df1["Trigger"] = df1["Trigger"].astype(str)
df1["Age"] = df1["Age"].astype(str)
df1["Lifestyle"] = df1["Lifestyle"].astype(str)
df1["Symptoms"] = df1["Symptoms"].astype(str)

# Save modified Dataset 1 to a new Excel file
df1.to_excel("external_triggers_lifestyle_rounded.xlsx", index=False)

print("Dataset 1 Severity after rounding:")
print(df1["Severity"].value_counts())

# =====================
# Dataset 2 preprocessing
# =====================
df2 = pd.read_csv("synthetic_allergy_data.csv.xls")

# Drop rows with missing Severity
df2 = df2.dropna(subset=["Severity"])

# Round Severity to nearest integer for classification
df2["Severity"] = df2["Severity"].round().astype(int)

# Ensure object columns are strings
for col in df2.select_dtypes(include=["object"]).columns:
    df2[col] = df2[col].astype(str)

# Save modified Dataset 2 to a new Excel file
df2.to_excel("synthetic_allergy_data_rounded.xlsx", index=False)

print("Dataset 2 Severity after rounding:")
print(df2["Severity"].value_counts())
