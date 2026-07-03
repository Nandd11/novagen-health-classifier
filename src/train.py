import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score
from preprocess import load_and_preprocess_data

def run_training():
    # Define the path to your dataset
    data_path = os.path.join("data", "patient_data.csv")
    target_col = "Target" 

    print("⏳ Loading data and executing preprocessing pipeline...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path, target_col)

    print("🌲 Initializing and training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    rec = recall_score(y_test, predictions, average='macro')

    print("\n🚀 Training Complete!")
    print(f"📊 Target Recall Achieved: {rec * 100:.1f}%")
    print("\n📋 Full Classification Report:")
    print(classification_report(y_test, predictions))

    # Save the models
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "random_forest_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    print("💾 Model and Scaler artifacts successfully saved to /models directory!")

if __name__ == "__main__":
    run_training()