import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Load datasets
# -------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Save test IDs
test_ids = test["id"]

# -------------------------
# Separate features and target
# -------------------------
y = train["health_condition"]

X = train.drop(columns=["id", "health_condition"])
X_test = test.drop(columns=["id"])

# -------------------------
# Fill missing values
# -------------------------
for col in X.columns:

    if X[col].dtype in ["int64", "float64"]:

        median = X[col].median()

        X[col] = X[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    else:

        mode = X[col].mode()[0]

        X[col] = X[col].fillna(mode)
        X_test[col] = X_test[col].fillna(mode)

# -------------------------
# Encode target
# -------------------------
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# -------------------------
# Encode categorical features
# -------------------------
for col in X.select_dtypes(include=["object", "string"]).columns:

    encoder = LabelEncoder()

    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)

    encoder.fit(combined)

    X[col] = encoder.transform(X[col].astype(str))
    X_test[col] = encoder.transform(X_test[col].astype(str))

# -------------------------
# Train model
# -------------------------
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

print("Training model...")

model.fit(X, y)

print("Training completed.")

# -------------------------
# Predict
# -------------------------
pred = model.predict(X_test)

pred = target_encoder.inverse_transform(pred)

# -------------------------
# Submission
# -------------------------
submission = pd.DataFrame({
    "id": test_ids,
    "health_condition": pred
})

submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")