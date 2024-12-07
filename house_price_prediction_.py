# Step 1: Import library
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Streamlit App
st.title("House Price Prediction")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv('https://github.com/YBIFoundation/Live-Projects/raw/main/IndiaHousePrice.csv')
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Step 2: Feature Selection
    st.write("### Define Features and Target Variable")
    X = data[['Number of bedrooms', 'Number of bathrooms', 'Living area sqft',
              'Number of schools nearby', 'Distance from the airport']]
    y = data['Price']

    # Step 3: Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2529)

    # Step 4: Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 5: Evaluate Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model Mean Squared Error (MSE): {mse:.2f}")

    # Step 6: Make Predictions
    st.write("### Make Predictions")
    bedrooms = st.number_input("Number of bedrooms", min_value=0, step=1)
    bathrooms = st.number_input("Number of bathrooms", min_value=0, step=1)
    living_area = st.number_input("Living area (sqft)", min_value=0, step=10)
    schools_nearby = st.number_input("Number of schools nearby", min_value=0, step=1)
    distance_airport = st.number_input("Distance from the airport (km)", min_value=0.0, step=0.1)

    if st.button("Predict Price"):
        input_data = pd.DataFrame({
            'Number of bedrooms': [bedrooms],
            'Number of bathrooms': [bathrooms],
            'Living area sqft': [living_area],
            'Number of schools nearby': [schools_nearby],
            'Distance from the airport': [distance_airport]
        })
        prediction = model.predict(input_data)[0]
        st.write(f"Predicted Price: ₹{prediction:,.2f}")
else:
    st.write("Please upload a dataset to proceed.")
