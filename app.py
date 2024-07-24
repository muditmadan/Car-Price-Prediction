import streamlit as st
import pandas as pd
import pickle

def load_model():
    with open('lasso_reg_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # Set the title and layout of the app
    st.set_page_config(page_title="Car Price Prediction App", layout="centered")
    st.title("🚗 Car Price Prediction App")
    st.write("Use this app to predict the selling price of a car based on various attributes.")

    # Load the model
    model = load_model()

    # Sidebar for inputs
    st.sidebar.header("Input Features")
    year = st.sidebar.text_input('Year of Purchase', value='2010')
    present_price = st.sidebar.text_input('Present Price of the car (in lakhs)', value='5.0')
    kms_driven = st.sidebar.text_input('Kms Driven', value='30000')
    fuel_type = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.sidebar.selectbox('Seller Type', ['Dealer', 'Individual'])
    transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.sidebar.selectbox('Owner Type', [0, 1, 2, 3])

    # Preprocess inputs
    try:
        year = int(year)
        present_price = float(present_price)
        kms_driven = int(kms_driven)
    except ValueError:
        st.sidebar.error("Please enter valid numbers for Year, Present Price, and Kms Driven.")
        st.stop()

    fuel_type_encoded = 0 if fuel_type == 'Petrol' else 1 if fuel_type == 'Diesel' else 2
    seller_type_encoded = 0 if seller_type == 'Dealer' else 1
    transmission_encoded = 0 if transmission == 'Manual' else 1

    # Display a button and make prediction
    if st.sidebar.button('Predict Selling Price'):
        input_data = pd.DataFrame([[year, present_price, kms_driven, fuel_type_encoded, seller_type_encoded, transmission_encoded, owner]], 
                                  columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])
        
        prediction = model.predict(input_data)
        st.write(f"### Predicted Selling Price: ₹ {prediction[0]:.2f} lakhs")

    # Footer
    st.write("---")
    st.write("App created by Mudit Madan")
    st.write("This model uses Lasso regression for predicting car prices.")

if __name__ == '__main__':
    main()
