import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Load the data
def preprocess_data():
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'data/product.csv')
    data = pd.read_csv(data_path)

    # 1. Handle Missing Values
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)

    # 2. Convert Date to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # 3. Feature Engineering
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Weekday'] = data['Date'].dt.day_name()

    # 4. Encode Weekday Column
    le_weekday = LabelEncoder()
    data['Weekday'] = le_weekday.fit_transform(data['Weekday'])
    
    # Display the mapping of days to numbers
    print("Weekday Encoding Mapping:", dict(zip(le_weekday.classes_, le_weekday.transform(le_weekday.classes_))))

    # 5. Encode Other Categorical Features
    categorical_cols = ['Category', 'Region', 'Weather Condition', 'Holiday/Promotion', 'Competitor Pricing', 'Seasonality']
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # 6. Scale Numerical Features
    numerical_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price', 'Discount']
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # Drop the original date column
    data.drop(columns=['Date'], inplace=True)

    # Save preprocessed data
    output_dir = os.path.join(current_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'preprocessed_product.csv')
    data.to_csv(output_path, index=False)
    print("Preprocessing completed. File saved at:", output_path)

# Call the function to run the preprocessing
if __name__ == '__main__':
    preprocess_data()
