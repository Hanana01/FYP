import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def train_model():
    # Load preprocessed dataset
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'data/preprocessed_product.csv')
    data = pd.read_csv(data_path)    

    # Specify the target column
    target_column = 'Units Sold'
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = os.path.join(current_dir, 'data/model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print("Model trained and saved successfully!")

def make_prediction(input_data):
    # Load the model
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'data/model.pkl')
    

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Train the model first.")
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    
    # Convert input data to DataFrame and ensure column order
    input_df = pd.DataFrame([input_data])
    preprocessed_data_path = os.path.join(current_dir, 'data/preprocessed_product.csv')
    data = pd.read_csv(preprocessed_data_path)
    input_df = input_df[data.drop(columns=['Units Sold']).columns]
    
    # Make prediction
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == '__main__':
    print("1. Train Model")
    print("2. Make Prediction")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        train_model()
    elif choice == '2':
        # Example input (replace with actual input values)
        input_data = {
            'Year': 2022,
            'Month': 5,
            'Weekday': 2, 
            'Category': 2,
            'Region': 1,
            'Weather Condition': 0,
            'Holiday/Promotion': 1,
            'Competitor Pricing': 3,
            'Seasonality': 2,
            'Inventory Level': -0.25,
            'Units Ordered': 0.15,
            'Demand Forecast': 0.45,
            'Price': -0.12,
            'Discount': 0.3
        }
        prediction = make_prediction(input_data)
        print(f"Predicted Units Sold: {prediction}")
    else:
        print("Invalid choice! Please select either 1 or 2.")
