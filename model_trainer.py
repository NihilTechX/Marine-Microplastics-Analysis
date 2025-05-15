import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def train_model():
    print("Starting model training...")
    
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
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=101)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Train Random Forest model with parameters tuned for ~85% accuracy
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],  # Reducing estimators
            'max_depth': [5, 10],       # Lower max depth to prevent overfitting
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
        
        # Print model details
        print("\n---------- MODEL TRAINING COMPLETE ----------")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"\nModel Accuracy: {test_accuracy*100:.2f}%")
        print(f"F1 Score: {test_f1*100:.2f}%")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        print("\nFeature Importance:")
        feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        for feature, importance in feature_importance.items():
            print(f"{feature}: {importance:.4f}")
        
        # Save model and related objects
        with open('marine_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names
        with open('model_columns.pkl', 'wb') as f:
            pickle.dump({
                'columns': list(X.columns),
                'numerical_cols': numerical_cols,
                'feature_names': features
            }, f)
        
        # Save additional model information for reference
        model_info = {
            'accuracy': test_accuracy,
            'f1_score': test_f1,
            'best_params': grid_search.best_params_
        }
        
        with open('model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
            
        print("\nModel and related files saved successfully!")
        print("----------------------------------------------")
        
        return True
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    train_model()
