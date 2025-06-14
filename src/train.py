from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.preprocessing import load_and_preprocess
import joblib
import os

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess()

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model to models/ directory
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/personality_model.pkl")
print("✅ Model saved to models/personality_model.pkl")
