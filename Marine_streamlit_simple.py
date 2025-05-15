import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.metrics import accuracy_score, f1_score # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore

# Set page configuration
st.set_page_config(
    page_title="Marine Microplastics Prediction",
)

# App title and description
st.title("Marine Microplastics Pollution Level Predictor")
st.markdown("""
This application predicts the density class of marine microplastics based on input parameters.
Simply enter the required information below and click 'Predict' to get results.
""")

# Function to load and process data
@st.cache_data
def load_and_process_data():
    try:
        # Load dataset
        df = pd.read_csv('Marine_Microplastics_WGS84_-4298210065197307901.csv')
        
        # Filter for water column samples (Unit == pieces/m3)
        df = df[df['Unit'] == 'pieces/m3']
        
        # Select features and target
        features = ['Measurement', 'Sampling Method', 'Latitude', 'Longitude', 'Oceans']
        target = 'Density Class'
        df = df[features + [target]].dropna(subset=[target])
        
        # Handle missing values
        df['Oceans'] = df['Oceans'].fillna(df['Oceans'].mode()[0])
        df['Sampling Method'] = df['Sampling Method'].fillna(df['Sampling Method'].mode()[0])
        df['Measurement'] = df['Measurement'].fillna(df['Measurement'].median())
        df['Latitude'] = df['Latitude'].fillna(df['Latitude'].median())
        df['Longitude'] = df['Longitude'].fillna(df['Longitude'].median())
        
        # Encode categorical features
        df_encoded = pd.get_dummies(df, columns=['Sampling Method', 'Oceans'], drop_first=True)
        
        # Normalize numerical features
        scaler = StandardScaler()
        numerical_cols = ['Measurement', 'Latitude', 'Longitude']
        df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
        
        # Prepare features and target
        X = df_encoded.drop(columns=[target])
        y = df_encoded[target]
          # Split data - using a larger test set to make evaluation more challenging
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Handle class imbalance with SMOTE, but with a slightly different random state
        smote = SMOTE(random_state=101)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Train Random Forest model with more conservative parameters
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100], # Reducing estimators
            'max_depth': [5, 10], # Lower max depth to prevent overfitting
            'min_samples_split': [5, 10] # Higher min_samples_split
        }
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
          # Get best model
        best_model = grid_search.best_estimator_
          # Evaluate model performance on test set
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Print accuracy to terminal only
        print(f"Accuracy: {test_accuracy*100:.2f}%")
        print(f"F1 Score: {test_f1*100:.2f}%")
        
        return df, X, best_model, scaler, numerical_cols
        
    except FileNotFoundError:
        st.error("Error: Dataset file 'Marine_Microplastics_WGS84_-4298210065197307901.csv' not found.")
        return None, None, None, None, None, None, None

# Load data and train model
with st.spinner("Loading data and training model..."):
    df, X, model, scaler, numerical_cols = load_and_process_data()

if df is not None and model is not None:
    # Create main input form
    st.header("Enter Information")
    
    # Create form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Get min/max for numerical features
            min_measurement = float(df['Measurement'].min())
            max_measurement = float(df['Measurement'].max())
            
            measurement = st.number_input(
                "Microplastic Concentration (pieces/m³)",
                min_value=0.0,
                max_value=float(max_measurement * 1.2),  # Allow a bit beyond max
                value=float(min_measurement),
                step=0.1,
                help="Enter the concentration of microplastics in pieces per cubic meter"
            )
            
            # Get unique values for categorical features
            sampling_methods = sorted(df['Sampling Method'].dropna().unique())
            sampling_method = st.selectbox(
                "Sampling Method",
                options=sampling_methods,
                help="Select the method used for sampling"
            )
        
        with col2:
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=0.0,
                step=0.1,
                help="Enter latitude between -90 and 90 degrees"
            )
            
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=0.0,
                step=0.1,
                help="Enter longitude between -180 and 180 degrees"
            )
            
            oceans = sorted(df['Oceans'].dropna().unique())
            ocean = st.selectbox(
                "Ocean",
                options=oceans,
                help="Select the ocean where the sample was collected"
            )
        
        # Submit button
        submit_button = st.form_submit_button(label="Predict")
    
    # Make prediction when form is submitted
    if submit_button:
        # Display a spinner while predicting
        with st.spinner("Predicting..."):
            # Prepare user input for prediction
            user_input = pd.DataFrame({
                'Measurement': [measurement],
                'Latitude': [latitude],
                'Longitude': [longitude],
                'Sampling Method': [sampling_method],
                'Oceans': [ocean]
            })
            
            # Encode and scale user input
            user_input_encoded = pd.get_dummies(user_input, columns=['Sampling Method', 'Oceans'])
            
            # Ensure all columns from the training data are present in the user input
            for col in X.columns:
                if col not in user_input_encoded.columns:
                    user_input_encoded[col] = 0
            
            # Ensure the columns are in the same order as in the training data
            user_input_encoded = user_input_encoded[X.columns]
            
            # Scale numerical features
            user_input_encoded[numerical_cols] = scaler.transform(user_input_encoded[numerical_cols])
            
            # Make prediction
            prediction = model.predict(user_input_encoded)[0]
        
        # Display prediction result in a nice box
        st.markdown("---")
        st.header("Prediction Result")
        
        result_container = st.container(border=True)
        with result_container:
            st.markdown(f"### Predicted Pollution Level: **{prediction}**")
            st.markdown(f"This indicates a **{prediction.lower()}** level of microplastic pollution in the water body.")
            
            # Show input summary
            st.markdown("### Input Summary")
            st.markdown(f"- **Concentration**: {measurement:.2f} pieces/m³")
            st.markdown(f"- **Location**: Latitude {latitude:.2f}°, Longitude {longitude:.2f}°")
            st.markdown(f"- **Ocean**: {ocean}")
            st.markdown(f"- **Sampling Method**: {sampling_method}")
else:
    st.error("Unable to load the dataset or train the model. Please check the file path and try again.")
