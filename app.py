import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import pickle
from tensorflow.keras import models
import joblib

def process_txt_file(file):
    if hasattr(file, 'getvalue'):  # Streamlit UploadedFile
        raw_bytes = file.getvalue()
    else:
        raw_bytes = file.read()

    # Attempt decoding with common encodings
    for encoding in ['utf-8', 'ISO-8859-1', 'utf-16', 'windows-1252']:
        try:
            content = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Could not decode file using common encodings.")

    # Split lines and find the start of data
    lines = content.splitlines()

    # Find the index of the second separator line
    data_start_index = None
    separator_count = 0

    for i, line in enumerate(lines):
        if line.startswith('---'):
            separator_count += 1
            if separator_count == 2:
                data_start_index = i + 1
                break

    if data_start_index is None:
        st.error("Data section not found in the uploaded file.")
        return pd.DataFrame(), pd.DataFrame()

    # Extract relevant data lines only
    data_lines = [
        line.strip().split()[:24] 
        for line in lines[data_start_index:] 
        if line.strip() and not line.startswith('#')
    ]
    
    columns = [
        "Routine Code", "Timestamp", "Routine Count", "Repetition Count", "Duration", "Integration Time [ms]",
        "Number of Cycles", "Saturation Index", "Filterwheel 1", "Filterwheel 2", "Zenith Angle [deg]", "Zenith Mode",
        "Azimuth Angle [deg]", "Azimuth Mode", "Processing Index", "Target Distance [m]",
        "Electronics Temp [°C]", "Control Temp [°C]", "Aux Temp [°C]", "Head Sensor Temp [°C]",
        "Head Sensor Humidity [%]", "Head Sensor Pressure [hPa]", "Scale Factor", "Uncertainty Indicator"
    ]
    
    df = pd.DataFrame(data_lines, columns=columns)
    
    # Parse the Timestamp column safely
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'].str.replace("T", " ").str.replace("Z", ""), 
        errors='coerce'
    )
    
    # Remove rows with invalid timestamps
    df = df[df['Timestamp'].notnull()].reset_index(drop=True)
    
    # Convert numeric columns properly
    numeric_columns = [col for col in df.columns if col not in ["Routine Code", "Timestamp"]]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # st.sidebar.write(f"After Conversion Column Types:\n{df.dtypes}")
    # st.sidebar.write(f"First Few Rows of DataFrame:\n{df.head()}")

    # Prepare the numeric DataFrame for scaling and model prediction
    df_numeric = df.drop(columns=["Routine Code", "Timestamp"], errors='ignore')

    return df, df_numeric



def load_and_preprocess_data(file, scaler):
    df, df_n = process_txt_file(file)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric = df_numeric.dropna(axis=1, how='all')

    if not df_numeric.empty:
        df_numeric = df_numeric.fillna(df_numeric.median(numeric_only=True))

    df_numeric = df_numeric.loc[:, df_numeric.nunique() > 1]

    st.sidebar.write(f"Column Types Before Scaling:\n{df.dtypes}")
    st.sidebar.write(f"Numeric Columns Found:\n{df_numeric.columns.tolist()}")
    st.sidebar.write(f"First Few Rows of Numeric Data:\n{df_numeric.head()}")

    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

    if df_numeric.empty:
        st.error("No valid numeric columns found. Please check the uploaded file.")
        raise ValueError("No valid numeric columns found in the uploaded file.")

    df_scaled = pd.DataFrame(
        scaler.transform(df_numeric),
        columns=df_numeric.columns,
        index=df.index
    )

    df_scaled["Timestamp"] = df["Timestamp"]

    return df, df_scaled

def test_anomaly_model(df_scaled, model):
    reconstructions = model.predict(df_scaled.drop(columns=["Timestamp"], errors='ignore'))
    reconstruction_errors = np.mean(np.abs(df_scaled.drop(columns=["Timestamp"], errors='ignore') - reconstructions), axis=1)
    
    threshold = np.percentile(reconstruction_errors, 99.9)
    df_scaled["Anomaly"] = (reconstruction_errors > threshold).astype(int)
    return df_scaled

def plot_data(df, df_scaled):
    columns_to_plot = [col for col in df.columns if col not in ["Timestamp", "Processed File"]]
    for column in columns_to_plot:
        fig = px.scatter(df, x="Timestamp", y=column, color=df_scaled["Anomaly"].map({0:'green', 1:'red'}),
                         color_discrete_map={"green": "green", "red": "red"},
                         title=f"{column} - Anomaly Detection")
        st.plotly_chart(fig)

def main():
    st.title("Deep Learning Anomaly Detection with Pre-trained Model")
    uploaded_file = st.file_uploader("Upload L0 file", type=["txt"])
    
    if uploaded_file is not None:
        try:
            # Load the pre-trained model with custom loss function
            loaded_autoencoder = models.load_model(
                'anomaly_model.h5', 
                custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
            )
            
            # Load the MinMaxScaler using joblib
            scaler = joblib.load('scaler.pkl')

            # Process the uploaded file
            df, df_numeric = process_txt_file(uploaded_file)
            
            # Ensure the Timestamp column is properly parsed
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
            
            # Retrieve expected feature names from the scaler
            expected_features = scaler.feature_names_in_
            
            # Align columns with expected features
            df_numeric = df_numeric.reindex(columns=expected_features, fill_value=0)
            
            # st.sidebar.write(f"Aligned Columns:\n{df_numeric.columns.tolist()}")

            # Scale the numeric data
            df_scaled = pd.DataFrame(scaler.transform(df_numeric), columns=df_numeric.columns)
            df_scaled["Timestamp"] = df["Timestamp"].reset_index(drop=True)
            
            # Predict anomalies using the loaded model
            reconstructions = loaded_autoencoder.predict(df_scaled.drop(columns=["Timestamp"], errors='ignore'))
            reconstruction_errors = np.mean(np.abs(df_scaled.drop(columns=["Timestamp"], errors='ignore') - reconstructions), axis=1)

            # Determine the anomaly threshold and classify anomalies
            threshold = np.percentile(reconstruction_errors, 90)
            df["Anomaly"] = (reconstruction_errors > threshold).astype(int)
            
            # Display anomalies in the Streamlit app
            anomalies = df[df["Anomaly"] == 1]
            st.write("Anomalies Detected:", anomalies)

            # Plotting the data with anomalies
            plot_data(df, df)

        except Exception as e:
            st.error(f"Error loading model or scaler: {e}")




if __name__ == "__main__":
    main()