import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# === Load Datasets ===
labeled_df = pd.read_csv("Crop_Data.csv")
unlabeled_df = pd.read_csv("crop_yield.csv")

# === Encode categorical columns in unlabeled ===
categorical_cols = ['Soil_Type', 'Crop_Type', 'Weather_Condition']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    unlabeled_df[col] = le.fit_transform(unlabeled_df[col])
    encoders[col] = le

# Convert Fertilizer_Used to int if it’s boolean
unlabeled_df['Fertilizer_Used'] = unlabeled_df['Fertilizer_Used'].astype(int)

# === Add missing columns in unlabeled ===
unlabeled_df['Estimated_Insects_Count'] = unlabeled_df['Rainfall_mm'].apply(lambda x: int(x * 2))
unlabeled_df['Pesticide_Use_Category'] = 3  # Assumed as "Currently Using"
unlabeled_df['Number_Doses_Week'] = 0
unlabeled_df['Number_Weeks_Used'] = 0
unlabeled_df['Number_Weeks_Quit'] = 0
unlabeled_df['Season'] = 1  # Default if unknown

# === Train Model ===
features = ['Estimated_Insects_Count', 'Crop_Type', 'Soil_Type',
            'Pesticide_Use_Category', 'Number_Doses_Week',
            'Number_Weeks_Used', 'Number_Weeks_Quit', 'Season']
X_train = labeled_df[features]
y_train = labeled_df['Crop_Damage']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Predict on Unlabeled ===
X_unlabeled = unlabeled_df[features]
predictions = model.predict(X_unlabeled)
unlabeled_df['Crop_Damage'] = predictions

# === Drop extra columns in unlabeled ===
columns_to_drop = ['Weather_Condition', 'Fertilizer_Used', 'Rainfall_mm']
unlabeled_df = unlabeled_df.drop(columns=columns_to_drop)

# === Ensure labeled and unlabeled have same columns and order ===
labeled_df = labeled_df.drop(columns=['ID']) if 'ID' in labeled_df.columns else labeled_df
combined_df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)

# === Save the combined dataset ===
combined_df.to_csv("combined_crop_damage_dataset.csv", index=False)
print("✅ Combined dataset saved as 'combined_crop_damage_dataset.csv'")
