import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

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

le_symptom = LabelEncoder()
df["Symptom"] = le_symptom.fit_transform(df["Symptom"])

# Save the symptom encoder to reuse in prediction
joblib.dump(le_symptom, "symptom_label_encoder.pkl")
print("Symptom LabelEncoder saved as symptom_label_encoder.pkl")

# =========================
# 3. Get all allergy types
# =========================

allergy_types = df["Allergy_Type"].unique()
print("\nAllergy types found:", allergy_types)

# =========================
# 4. Hyperparameter spaces
# =========================

rf_param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

gb_param_dist = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4],
    "subsample": [0.8, 1.0]
}

lr_param_dist = {
    "C": np.logspace(-2, 2, 10),
    "penalty": ["l2"],
    "solver": ["lbfgs"],
}

# =========================
# 5. Loop through each allergy type
# =========================

for allergy in allergy_types:
    print(f"\n================= {allergy} =================")

    # Filter dataset for this allergy type
    subset = df[df["Allergy_Type"] == allergy].copy()

    # Features & Target
    X = subset.drop(["Severity", "Allergy_Type"], axis=1)
    y = subset["Severity"]

    print(f"Subset shape for {allergy}: {X.shape}")

    # =========================
    # 5a. Scale features
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_filename = f"{allergy.lower().replace(' ', '_')}_scaler.pkl"
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved as {scaler_filename}")

    # =========================
    # 5b. Train-test split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Dictionary to store best tuned models
    tuned_models = {}

    # =========================
    # 6. Tune Logistic Regression
    # =========================
    print(f"\n--- Tuning LogisticRegression for {allergy} ---")
    lr_base = LogisticRegression(max_iter=1000, random_state=42)

    lr_search = RandomizedSearchCV(
        estimator=lr_base,
        param_distributions=lr_param_dist,
        n_iter=10,
        scoring="accuracy",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    lr_search.fit(X_train, y_train)
    best_lr = lr_search.best_estimator_
    tuned_models["LogisticRegression"] = best_lr

    print("Best LogisticRegression params:", lr_search.best_params_)
    y_lr_pred = best_lr.predict(X_test)
    lr_acc = accuracy_score(y_test, y_lr_pred)
    print(f"LogisticRegression Tuned Accuracy: {lr_acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_lr_pred))

    lr_filename = f"{allergy.lower().replace(' ', '_')}_logisticregression_tuned.pkl"
    joblib.dump(best_lr, lr_filename)
    print(f"Tuned LogisticRegression saved as {lr_filename}")

    # =========================
    # 7. Tune RandomForest
    # =========================
    print(f"\n--- Tuning RandomForest for {allergy} ---")
    rf_base = RandomForestClassifier(random_state=42)

    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_param_dist,
        n_iter=20,
        scoring="accuracy",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    tuned_models["RandomForest"] = best_rf

    print("Best RandomForest params:", rf_search.best_params_)
    y_rf_pred = best_rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_rf_pred)
    print(f"RandomForest Tuned Accuracy: {rf_acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_rf_pred))

    rf_filename = f"{allergy.lower().replace(' ', '_')}_randomforest_tuned.pkl"
    joblib.dump(best_rf, rf_filename)
    print(f"Tuned RandomForest saved as {rf_filename}")

    # =========================
    # 8. Tune GradientBoosting
    # =========================
    print(f"\n--- Tuning GradientBoosting for {allergy} ---")
    gb_base = GradientBoostingClassifier(random_state=42)

    gb_search = RandomizedSearchCV(
        estimator=gb_base,
        param_distributions=gb_param_dist,
        n_iter=20,
        scoring="accuracy",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    gb_search.fit(X_train, y_train)
    best_gb = gb_search.best_estimator_
    tuned_models["GradientBoosting"] = best_gb

    print("Best GradientBoosting params:", gb_search.best_params_)
    y_gb_pred = best_gb.predict(X_test)
    gb_acc = accuracy_score(y_test, y_gb_pred)
    print(f"GradientBoosting Tuned Accuracy: {gb_acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_gb_pred))

    gb_filename = f"{allergy.lower().replace(' ', '_')}_gradientboosting_tuned.pkl"
    joblib.dump(best_gb, gb_filename)
    print(f"Tuned GradientBoosting saved as {gb_filename}")

    # =========================
    # 9. Build Stacking Ensemble from tuned models
    # =========================
    print(f"\n--- Training Tuned Stacking Ensemble for {allergy} ---")

    stack_estimators = [
        ("rf", tuned_models["RandomForest"]),
        ("gb", tuned_models["GradientBoosting"]),
        ("lr", tuned_models["LogisticRegression"]),
    ]

    meta_learner = LogisticRegression(max_iter=1000, random_state=42)

    stacking_model = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=meta_learner,
        passthrough=False,
        n_jobs=-1
    )

    stacking_model.fit(X_train, y_train)
    y_stack_pred = stacking_model.predict(X_test)

    stack_acc = accuracy_score(y_test, y_stack_pred)
    print(f"Tuned Stacking Ensemble Accuracy: {stack_acc:.4f}")
    print("Tuned Stacking Ensemble Classification Report:")
    print(classification_report(y_test, y_stack_pred))

    stack_filename = f"{allergy.lower().replace(' ', '_')}_stacking_tuned.pkl"
    joblib.dump(stacking_model, stack_filename)
    print(f"Tuned Stacking Ensemble saved as {stack_filename}")

print("\n✅ Hyperparameter tuning and training complete for all allergy types.")
