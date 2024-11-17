import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle
from preprocess import preprocess_data, print_data_info

def train_model():
    # 1. Preprocess the data with specified columns
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data(
        '../data/wearable_health_data.csv'
    )

    # Print information about the preprocessed data
    print_data_info(X_train, y_train)  # Updated to match new signature

    # Get number of classes
    n_classes = len(np.unique(y_train))
    print(f"Number of unique classes: {n_classes}")

    # The rest of the function remains exactly the same
    if n_classes == 2:
        objective = 'binary:logistic'
    else:
        objective = 'multi:softprob'

    model = xgb.XGBClassifier(
        objective=objective,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 0.9],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=2,
        error_score='raise'
    )

    try:
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        
        print("Training shape:", X_train.shape)
        print("Target shape:", y_train.shape)
        print("Unique classes in target:", np.unique(y_train))
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        model_artifacts = {
            'model': best_model,
            'scaler': scaler,
            'encoders': encoders,
            'feature_names': X_train.columns.tolist(),
            'best_params': grid_search.best_params_,
            'n_classes': n_classes
        }

        with open('../models/disease_predictor.pkl', 'wb') as file:
            pickle.dump(model_artifacts, file)

        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.int32)
        y_pred = best_model.predict(X_test)

        print("\nModel Evaluation:")
        print("----------------")
        print("Best parameters:", grid_search.best_params_)
        print("\nAccuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        })
        print("\nTop 10 Most Important Features:")
        print(feature_importance.sort_values('importance', ascending=False).head(10))

        return best_model, scaler, encoders

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        print("\nDataset information:")
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_train dtype:", X_train.dtypes)
        print("y_train unique values:", np.unique(y_train))
        raise

if __name__ == "__main__":
    train_model()
