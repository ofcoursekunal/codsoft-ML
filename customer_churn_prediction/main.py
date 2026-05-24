import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv("churn.csv")

# Show dataset
print(df.head())

# Remove unnecessary columns
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# Convert categorical columns into numbers
categorical_columns = ["Geography", "Gender"]

label_encoder = LabelEncoder()

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Input and Output
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred, zero_division=1))

# Custom prediction
sample = X_test.iloc[0:1]

prediction = model.predict(sample)

print("\nPrediction:", prediction)

if prediction[0] == 1:
    print("Customer will leave")
else:
    print("Customer will stay")