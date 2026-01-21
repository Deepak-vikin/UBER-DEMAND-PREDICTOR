import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def evaluate():
    # 1. Load Data
    try:
        df = pd.read_csv('taxi.csv')
    except FileNotFoundError:
        print("Error: taxi.csv not found.")
        return

    # 2. Define Features and Target
    # Based on previous file reads, columns are:
    # Priceperweek,Population,Monthlyincome,Averageparkingpermonth,Numberofweeklyriders
    X = df[['Priceperweek', 'Population', 'Monthlyincome', 'Averageparkingpermonth']]
    y = df['Numberofweeklyriders']

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 4. Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Make Predictions
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0) # Apply the same non-negative constraint as in the app

    # 6. Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

   

    print("-" * 30)
    print("MODEL PERFORMANCE METRICS")
    print("-" * 30)
    print(f"Mean Squared Error (MSE): {mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()
