'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =======================
# Load the dataset
# =======================
file_path = "allergies_clean.csv"   # <--- keep your CSV in the same folder or give full path
df = pd.read_csv(file_path)

# Clean column names (remove extra spaces or hidden chars)
df.columns = df.columns.str.strip()

# =======================
# Encode categorical columns
# =======================
categorical_cols = ['Cause', 'Allergy_type', 'CATEGORY', 'Symptoms']
for col in categorical_cols:
    df[col] = df[col].astype(str)   # Convert to string to avoid NaN issues
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# =======================
# Split features and target
# =======================
X = df.drop('severity', axis=1)
y = df['severity']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split (stratify keeps severity class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =======================
# Define models
# =======================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# =======================
# Train & Evaluate
# =======================
print("\n===== Model Accuracies =====")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Best model
best_model = max(results, key=results.get)
print(f"\n✅ Best Model: {best_model} with accuracy {results[best_model]:.4f}")
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =======================
# Load the dataset
# =======================
file_path = "allergies_clean.csv"   # <-- keep your CSV in the same folder or give full path
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  # clean column names

# =======================
# Encode categorical columns
# =======================
categorical_cols = ['Cause', 'Allergy_type', 'CATEGORY', 'Symptoms']
for col in categorical_cols:
    df[col] = df[col].astype(str)  # Convert to string to avoid NaN issues
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# =======================
# Split features and target
# =======================
X = df.drop('severity', axis=1)
y = df['severity']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split (stratify keeps severity class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =======================
# Define models
# =======================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

# =======================
# Train, Evaluate & Save Models
# =======================
print("\n===== Model Accuracies =====")
results = {}
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"Accuracy: {acc:.4f}")
    
    # Detailed classification report
    print(classification_report(y_test, y_pred))
    
    # Save model as .pkl
    filename = f"{name.lower()}.pkl"
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# Best model
best_model = max(results, key=results.get)
print(f"\n✅ Best Model: {best_model} with accuracy {results[best_model]:.4f}")

