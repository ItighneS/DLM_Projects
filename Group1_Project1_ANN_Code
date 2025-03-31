import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Streamlit UI
st.title('📊 Loan Prediction Dashboard - Deep Learning')

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload CSV File", type=['csv'])

if uploaded_file is not None:
    # Load Data
    data = pd.read_csv(uploaded_file)
    st.write("### 🔍 Dataset Preview:", data.head())

    # Preprocessing
    if 'Loan_ID' in data.columns:
        data.drop(columns=['Loan_ID'], inplace=True)

    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)

    # Encoding categorical variables
    encoding_dict = {
        'Gender': {'Male': 0, 'Female': 1},
        'Married': {'No': 0, 'Yes': 1},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Loan_Status': {'N': 0, 'Y': 1}
    }
    for col, mapping in encoding_dict.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)

    # Normalize Numerical Data
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in numerical_cols:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    # Define Features and Target
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    
    # Sidebar for Hyperparameter Selection
    st.sidebar.header("Model Hyperparameters")

    # Number of layers
    num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)

        # Create activation function selection for each layer
    activation_functions = []
    neurons_per_layer = []

    for i in range(num_layers):
        neurons = st.sidebar.slider(f"Neurons in Layer {i+1}", 1, 100, 10)
        activation = st.sidebar.selectbox(f"Activation for Layer {i+1}", ['relu', 'tanh', 'sigmoid'], index=0)
            
        neurons_per_layer.append(neurons)
        activation_functions.append(activation)

    optimizer = st.sidebar.selectbox("Optimizer", ['adam', 'sgd', 'rmsprop'])
    loss_function = st.sidebar.selectbox("Loss Function", ['binary_crossentropy', 'mean_squared_error', 'hinge'])
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
    epochs = st.sidebar.slider("Number of Epochs", 1, 100, 50)

    # Model Training Button
    if st.button("🚀 Train Model"):
        # Build ANN Model
        
        model = Sequential()
        model.add(InputLayer(input_shape=(X_train.shape[1],)))

        for i in range(num_layers):
            model.add(Dense(units=neurons_per_layer[i], activation=activation_functions[i]))
            model.add(Dropout(dropout_rate))  # Dropout to prevent overfitting

        # Output layer (Binary Classification -> Sigmoid)
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile Model
        model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])


        st.write(f"🔧 Using Optimizer: {optimizer}")
        st.write(f"🔧 Using Activation: {activation}")
        st.write(f"🔧 Neurons per layer: {neurons_per_layer}")
        st.write(f"🔧 Dropout Rate: {dropout_rate}")
        st.write(f"🔧 Epochs: {epochs}")


        # Train Model
        with st.spinner('Training ANN Model... Please wait!'):
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=1)

        st.success("🎉 Model Training Completed!")

        # Plot Loss
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

        # Plot Accuracy
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Training Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)

        # Final Evaluation
        loss, acc = model.evaluate(X_test, y_test)
        st.write(f"### 📈 Final Validation Accuracy: {acc:.4f}")
