import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# function for the status prediction
def status(x, y, status_features):
    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier on the training data
    rf_classifier.fit(x, y)

    # Predict the target variable for the test set
    statuss = rf_classifier.predict(status_features)

    st.write('The predicted status is: ', 'Won' if statuss == 1 else 'Lost')


# function for the selling price prediction
def sp(f, t, sp_features):
    # Create a Random Forest Regressor with specified parameters
    random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model on the training data
    random_forest_regressor.fit(f, t)

    # Predict the target variable for the x values
    s_price = random_forest_regressor.predict(sp_features)

    st.write('The predicted selling price is: ', s_price)

# main function to run streamlit
def main():
    # read csv file
    df = pd.read_csv('preprocessed.csv')
    # setting page configuration
    st.set_page_config(page_title="Industrial_Copper_Modelling", layout='wide', initial_sidebar_state='collapsed')
    # using sidebar for features
    st.sidebar.title('FEATURES')
    country = st.sidebar.selectbox('Select country code', df['country'].unique())
    item_type = st.sidebar.selectbox('Select item type', ['W', 'S', 'Others', 'PL', 'WI', 'IPL'])
    application = st.sidebar.selectbox('Select application', df['application'].unique())
    selling_price = st.sidebar.number_input('Enter the selling price', value=df['selling_price'].mean())
    days_delivery = st.sidebar.number_input('Enter the days to deliver', value=50)
    quantity = st.sidebar.number_input('Enter the quantity in tons', value=50)
    thickness = st.sidebar.number_input('Enter the thickness', value=5)
    width = st.sidebar.number_input('Enter the width', value=5)

    # Define a mapping of categories to numeric values
    category_mapping = {'W': 0,
                        'S': 1,
                        'Others': 2,
                        'PL': 3,
                        'WI': 4,
                        'IPL': 5,
                        'SLAWR': 6}

    # Centered heading using st.markdown
    st.markdown('<h1 style="text-align: center;">INDUSTRIAL COPPER MODELLING</h1>', unsafe_allow_html=True)
    st.header('STATUS')
    # dataframe for provided features
    status_data = {'country': country,
                   'item type': category_mapping[item_type],
                   'application': application,
                   'selling_price': selling_price,
                   'duration': days_delivery,
                   'log_quantity': np.log(quantity),
                   'log_thickness': np.log(thickness),
                   'log_width': np.log(width)}
    status_features = pd.DataFrame(status_data, index=[0])

    # Separate the target column 'status' (y) and feature columns (X)
    x = df.drop('status', axis=1)  # Drop the 'status' column to create the feature matrix
    y = df['status']  # Assign the 'status' column to the target variable y

    if st.button('predict status'):
        # calling function to predict status
        status(x, y, status_features)

    st.header('SELLING PRICE')
    # dataframe for provided features
    sp_data = {'country': country,
               'application': application,
               'duration': days_delivery,
               'log_quantity': np.log(quantity),
               'log_thickness': np.log(thickness),
               'log_width': np.log(width)}
    sp_features = pd.DataFrame(sp_data, index=[0])

    df_regress = df.drop(['status', 'item type'], axis=1)
    # Assuming 'selling price' is your target variable and the rest are features
    t = df_regress['selling_price']  # This creates the target variable t
    f = df_regress.drop('selling_price',
                        axis=1)  # This creates the feature matrix f by dropping the 'selling price' column
    if st.button('Predict selling price'):
        # calling function to predict selling price
        sp(f, t, sp_features)


if __name__ == "__main__":
    # This block will be executed when the script is run as the main program
    main()
