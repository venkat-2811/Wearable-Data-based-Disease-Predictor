import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(filepath):
    # Load data
    data = pd.read_csv(filepath)
    
    # Print initial data info
    print("Initial data info:")
    print(data['Disease_Label'].describe())
    
    # 1. Handle missing values
    # For numerical columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col != 'Disease_Label':  # Skip the target column
            # Replace missing values with median of non-missing values
            data[col].fillna(data[col].median(), inplace=True)
    
    # For Activity_Level (categorical)
    data['Activity_Level'].fillna(data['Activity_Level'].mode()[0], inplace=True)
    
    # 2. Encode Activity_Level
    label_encoder_activity = LabelEncoder()
    data['Activity_Level'] = label_encoder_activity.fit_transform(data['Activity_Level'])
    
    # 3. Process Disease_Label
    # Ensure Disease_Label is properly processed for classification
    disease_labels = data['Disease_Label'].unique()
    print(f"\nUnique disease labels before processing: {disease_labels}")
    
    # If Disease_Label contains float values, convert them to integers
    if data['Disease_Label'].dtype == float:
        # Fill NaN values with median before converting to int
        data['Disease_Label'] = data['Disease_Label'].fillna(data['Disease_Label'].median())
        # Round to nearest integer if they're continuous values
        data['Disease_Label'] = data['Disease_Label'].round().astype(int)
    
    # Encode the labels to ensure they start from 0 and are consecutive
    label_encoder_disease = LabelEncoder()
    data['Disease_Label'] = label_encoder_disease.fit_transform(data['Disease_Label'])
    
    print(f"Unique disease labels after processing: {np.unique(data['Disease_Label'])}")
    
    # 4. Split features and target
    X = data.drop('Disease_Label', axis=1)
    y = data['Disease_Label']
    
    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # 7. Create a dictionary of encoders for future use
    encoders = {
        'activity': label_encoder_activity,
        'disease': label_encoder_disease
    }
    
    # Print final shapes and information
    print("\nFinal preprocessed data shapes:")
    print(f"X_train: {X_train_scaled.shape}")
    print(f"X_test: {X_test_scaled.shape}")
    print(f"Unique classes in y_train: {np.unique(y_train)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders
    
def print_data_info(X_train, y_train):
    """Print information about the preprocessed data"""
    print("\nDetailed Data Information:")
    print("--------------------------")
    print(f"Training set shape: {X_train.shape}")
    print(f"Feature names: {X_train.columns.tolist()}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of training samples: {X_train.shape[0]}")
    print("\nTarget variable information:")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Unique classes: {np.unique(y_train)}")
    print("\nClass distribution in training set:")
    print(pd.Series(y_train).value_counts().sort_index())