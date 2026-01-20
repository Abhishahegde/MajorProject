import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

FILE_PATH = "dust_pet_pollen.csv"

# ======= USER: CHANGE THIS INDEX TO TEST DIFFERENT ROWS =======
ROW_INDEX_TO_TEST = 3   # 3 = your sample from the head output
# ===============================================================

def main():
    # 1) Load raw dataset
    df = pd.read_csv(FILE_PATH)

    print("Loaded dataset with shape:", df.shape)

    # 2) Load the saved Symptom LabelEncoder
    le_symptom = joblib.load("symptom_label_encoder.pkl")
    print("Loaded Symptom LabelEncoder.")

    # 3) Encode Symptom column using the SAME encoder as training
    #    Important: use transform, NOT fit_transform
    df["Symptom"] = le_symptom.transform(df["Symptom"])

    # 4) Pick the sample row by index
    if ROW_INDEX_TO_TEST < 0 or ROW_INDEX_TO_TEST >= len(df):
        raise ValueError(f"ROW_INDEX_TO_TEST {ROW_INDEX_TO_TEST} is out of range 0..{len(df)-1}")

    sample_row = df.iloc[ROW_INDEX_TO_TEST].copy()  # Series

    print("\n=== Sample row from CSV ===")
    print(sample_row)

    allergy_type = sample_row["Allergy_Type"]
    true_severity = sample_row["Severity"]

    print(f"\nAllergy_Type: {allergy_type}")
    print(f"True Severity: {true_severity}")

    # 5) Prepare features in SAME WAY as training
    #    Drop Severity and Allergy_Type
    feature_cols = df.columns.drop(["Severity", "Allergy_Type"])
    X_all = df[feature_cols]  # full feature matrix (for scaler shape)

    # Extract this sample's features as DataFrame
    X_sample = sample_row[feature_cols].to_frame().T   # shape (1, n_features)

    # 6) Load the scaler for this allergy type
    scaler_filename = f"{allergy_type.lower().replace(' ', '_')}_scaler.pkl"
    scaler = joblib.load(scaler_filename)
    print(f"\nLoaded scaler: {scaler_filename}")

    # Scale the sample
    X_sample_scaled = scaler.transform(X_sample)

    # 7) Load the stacking model for this allergy type
    #model_filename = f"{allergy_type.lower().replace(' ', '_')}_stacking.pkl"
    model_filename = f"{allergy_type.lower().replace(' ', '_')}_stacking_tuned.pkl"

    model = joblib.load(model_filename)
    print(f"Loaded stacking model: {model_filename}")

    # 8) Predict severity
    pred_severity = model.predict(X_sample_scaled)[0]

    print("\n=== Prediction Result ===")
    print(f"True Severity:      {true_severity}")
    print(f"Predicted Severity: {pred_severity}")
    print("Correct? ->", pred_severity == true_severity)

    # (Optional) Map severities to labels for nice printing
    severity_map = {
        1: "Very Mild",
        2: "Mild",
        3: "Moderate",
        4: "Severe",
        5: "Very Severe"
    }

    if true_severity in severity_map:
        print(f"\nTrue Severity Label:      {severity_map[true_severity]}")
    if pred_severity in severity_map:
        print(f"Predicted Severity Label: {severity_map[pred_severity]}")

if __name__ == "__main__":
    main()
