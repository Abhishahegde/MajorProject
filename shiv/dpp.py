import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =========================
# 1. Load dataset
# =========================

FILE_PATH = "dust_pet_pollen.csv"  # make sure this file is in the same folder

df = pd.read_csv(FILE_PATH)

print("Dataset loaded.")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# =========================
# 2. Encode 'Symptom' column
# =========================
# Current approach: treat each unique symptom combination as a category

le_symptom = LabelEncoder()
df["Symptom"] = le_symptom.fit_transform(df["Symptom"])

# (Optional) Save the encoder if you want to use it later for prediction scripts
joblib.dump(le_symptom, "symptom_label_encoder.pkl")
print("Symptom LabelEncoder saved as symptom_label_encoder.pkl")

# =========================
# 3. Get all allergy types
# =========================

allergy_types = df["Allergy_Type"].unique()
print("\nAllergy types found:", allergy_types)

# =========================
# 4. Loop through each allergy type
# =========================

for allergy in allergy_types:
    print(f"\n================= {allergy} =================")

    # Filter dataset for this allergy type
    subset = df[df["Allergy_Type"] == allergy].copy()

    # Features & Target
    # We drop 'Severity' (target) and 'Allergy_Type' (we're working per allergy)
    X = subset.drop(["Severity", "Allergy_Type"], axis=1)
    y = subset["Severity"]

    print(f"Subset shape for {allergy}: {X.shape}")

    # =========================
    # 5. Scale features
    # =========================
    # Note: This fits scaler on all X (same as your original code)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # (Optional) Save scaler per allergy type if you want consistent scaling later
    scaler_filename = f"{allergy.lower().replace(' ', '_')}_scaler.pkl"
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved as {scaler_filename}")

    # =========================
    # 6. Train-test split
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # 7. Base models
    # =========================

    base_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    # Train and evaluate each base model separately (optional but useful for comparison)
    for name, model in base_models.items():
        print(f"\n--- Training {name} for {allergy} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Save each base model
        filename = f"{allergy.lower().replace(' ', '_')}_{name.lower()}.pkl"
        joblib.dump(model, filename)
        print(f"Model saved as {filename}")

    # =========================
    # 8. Stacking Ensemble
    # =========================
    # Ensemble of RandomForest + GradientBoosting + LogisticRegression
    # Meta-learner: Logistic Regression

    print(f"\n--- Training Stacking Ensemble for {allergy} ---")

    stack_estimators = [
        ("rf", RandomForestClassifier(random_state=42)),
        ("gb", GradientBoostingClassifier(random_state=42)),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ]

    meta_learner = LogisticRegression(max_iter=1000, random_state=42)

    stacking_model = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=meta_learner,
        passthrough=False,  # if True, adds original features along with base preds
        n_jobs=-1
    )

    stacking_model.fit(X_train, y_train)
    y_stack_pred = stacking_model.predict(X_test)

    stack_acc = accuracy_score(y_test, y_stack_pred)
    print(f"Stacking Ensemble Accuracy: {stack_acc:.4f}")
    print("Stacking Ensemble Classification Report:")
    print(classification_report(y_test, y_stack_pred))

    # Save stacking model
    stack_filename = f"{allergy.lower().replace(' ', '_')}_stacking.pkl"
    joblib.dump(stacking_model, stack_filename)
    print(f"Stacking Ensemble saved as {stack_filename}")

print("\n✅ Training complete for all allergy types.")
