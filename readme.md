INDUSTRIAL COPPER MODELLING

Overview

The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.

Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer. You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.

This project aims to provide a comprehensive toolkit for the copper industry, addressing these challenges through machine learning techniques.
Key Features

    Exploring skewness and outliers in the dataset.
    Transforming the data into a suitable format and performing necessary cleaning and pre-processing steps.
    Building a machine learning regression model which predicts the continuous variable 'Selling_Price'.
    Developing a machine learning classification model which predicts the Status: WON or LOST.
    Creating a Streamlit web application where users can input each column value and receive the predicted Selling_Price or Status (Won/Lost).

    Usage

    Explore and preprocess your data by running the data preprocessing scripts.

    Train and evaluate the machine learning regression and classification models using the provided Jupyter notebooks.

    Run the Streamlit web application to interact with the models and get predictions.

    Project Structure

The project is organized as follows:

    data/: Contains the dataset used for training and testing.
    notebooks/: Jupyter notebooks for data exploration, preprocessing, and model building.
    src/: Python scripts for data preprocessing and utility functions.
    app.py: Streamlit web application for user interaction.

    License

This project is licensed under the MIT License - see the LICENSE.md file for details.
