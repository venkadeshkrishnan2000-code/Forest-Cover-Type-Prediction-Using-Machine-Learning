import pandas as pd
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ FIRST: Load data
df = pd.read_csv("dataforest.csv")

# ✅ THEN: Use df
df.hist(figsize=(10,8))
plt.show()

sns.countplot(x='Cover_Type', data=df)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

os.makedirs("model", exist_ok=True)

#data
df = pd.read_csv("dataforest.csv")
print("\n📊 Dataset Shape:")
print(df.shape)

print("\n📋 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe())

print("\n🌲 Cover Type Distribution:")
print(df['Cover_Type'].value_counts())
# ✅ Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode
target_encoder = LabelEncoder()
df['Cover_Type'] = target_encoder.fit_transform(df['Cover_Type'])

joblib.dump(target_encoder, "model/target_encoder.pkl")

#Split data
X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "model/scaler.pkl")

#Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Handle imbalance
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

#Train model (simple & stable)
#Train model (improved)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

#Save model
joblib.dump(model, "model/model.pkl")

print("✅ ALL FILES SAVED SUCCESSFULLY!")
print("Files inside model folder:", os.listdir("model"))