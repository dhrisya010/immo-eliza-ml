import joblib
import numpy as np

def load_model(filepath):
    """Load a saved model from file."""
    return joblib.load(filepath)

def predict_price(model, features):
    """Predict house price given a model and feature values."""
    features_array = np.array(features).reshape(1, -1)  # Reshape for a single sample
    return model.predict(features_array)[0]

if __name__ == "__main__":
    # Load the saved model
    model = load_model('house_price_model.pkl')
    print("Model loaded from house_price_model.pkl")

    # Define features for a new house (replace with actual feature names and values)
    features = [120, 3, 2, 1, 10]  # Example: [area, bedrooms, bathrooms, garages, age]
    
    # Predict price
    predicted_price = predict_price(model, features)
    print(f"Predicted house price: {predicted_price}")


