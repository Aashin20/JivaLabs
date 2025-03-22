import pandas as pd
import numpy as np
from collections import Counter
import os
import joblib
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from skimpy import skim
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


from sklearn.metrics import (accuracy_score, 
                           confusion_matrix, 
                           precision_recall_curve, 
                           roc_curve)


SEED = 42
TEST_SIZE = 0.3

def load_data(filepath):
    """Load and preprocess the dataset"""
    data = pd.read_csv(filepath)
    data = data.drop('ID', axis=1)
    

    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.replace(" ", "_")
    data.columns = data.columns.str.replace('[{}]', '', regex=True)
    
    return data

def prepare_data(data):
    """Split data into features and target"""
    X = data.drop('Diagnosis_(ALS)', axis=1)
    y = data['Diagnosis_(ALS)']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=SEED, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):
    """Preprocess the features"""
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.to_list()
    
    transformers = [
        ('ohe', 
         OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         categorical_features)
    ]
    
    preprocessor = ColumnTransformer(
        transformers, 
        remainder='passthrough', 
        verbose_feature_names_out=False
    ).set_output(transform='pandas')
    
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    return X_train_prep, X_test_prep, preprocessor

def train_models(X_train, y_train, X_test, y_test):
    """Train and evaluate multiple models"""
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=SEED),
        'ExtraTree': ExtraTreeClassifier(random_state=SEED),
        'RandomForest': RandomForestClassifier(random_state=SEED, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(random_state=SEED, bootstrap=True, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=SEED, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=SEED, verbosity=-1),
        'CatBoost': CatBoostClassifier(random_state=SEED, verbose=0)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {'train_acc': train_acc, 'test_acc': test_acc}
        print(f"{name} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    return results

def save_model(model, preprocessor, model_name="catboost", base_path="models"):
    """
    Save model and preprocessor with timestamp
    Returns the paths of saved files
    """
   
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
   
    model_path = os.path.join(base_path, f"{model_name}_model_{timestamp}.pkl")
    preprocessor_path = os.path.join(base_path, f"{model_name}_preprocessor_{timestamp}.pkl")
    
   
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    return model_path, preprocessor_path

def train_best_model(X_train, y_train, X_test, y_test):
    """
    Train the best model (CatBoost) and return model performance metrics
    """
    model = CatBoostClassifier(random_state=SEED, verbose=0)
    model.fit(X_train, y_train)
    
  
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_prob_train = model.predict_proba(X_train)[:, 1]
    y_pred_prob_test = model.predict_proba(X_test)[:, 1]
    
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_probabilities': y_pred_prob_train,
        'test_probabilities': y_pred_prob_test
    }
    
    return model, metrics

def plot_results(model, X_train, y_train, X_test, y_test):
    """Plot confusion matrix and ROC curves"""

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_prob_train = model.predict_proba(X_train)[:, 1]
    y_pred_prob_test = model.predict_proba(X_test)[:, 1]
    
   
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    
    sns.heatmap(conf_matrix_train, annot=True, fmt='d', ax=ax[0])
    ax[0].set_title('Train Confusion Matrix')
    
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', ax=ax[1])
    ax[1].set_title('Test Confusion Matrix')
    
    plt.tight_layout()
    plt.show()
    
    
    pr_train, rec_train, _ = precision_recall_curve(y_train, y_pred_prob_train)
    pr_test, rec_test, _ = precision_recall_curve(y_test, y_pred_prob_test)
    
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_prob_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_prob_test)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    
    ax[0].plot(rec_train, pr_train, label='Train')
    ax[0].plot(rec_test, pr_test, label='Test')
    ax[0].set_title('Precision-Recall Curve')
    ax[0].legend()
    
    ax[1].plot(fpr_train, tpr_train, label='Train')
    ax[1].plot(fpr_test, tpr_test, label='Test')
    ax[1].set_title('ROC Curve')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the training pipeline
    Returns trained model file paths and performance metrics
    """
    try:
     
        print("Loading data...")
        data = load_data('../Datasets/Minsk2020_ALS_dataset.csv')
        
     
        print("Preparing data...")
        X_train, X_test, y_train, y_test = prepare_data(data)
        
        
        print("Preprocessing data...")
        X_train_prep, X_test_prep, preprocessor = preprocess_data(X_train, X_test)
        

        print("Training all models...")
        results = train_models(X_train_prep, y_train, X_test_prep, y_test)
        
       
        print("Training best model (CatBoost)...")
        best_model, metrics = train_best_model(X_train_prep, y_train, X_test_prep, y_test)
        
     
        print("Saving model and preprocessor...")
        model_path, preprocessor_path = save_model(best_model, preprocessor)
        
    
        print("Generating plots...")
        plot_results(best_model, X_train_prep, y_train, X_test_prep, y_test)
        
        return {
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'metrics': metrics,
            'all_models_results': results
        }
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
   
    results = main()
    
   
    print("\nTraining Results:")
    print(f"Model saved at: {results['model_path']}")
    print(f"Preprocessor saved at: {results['preprocessor_path']}")
    print(f"\nModel Performance:")
    print(f"Train Accuracy: {results['metrics']['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['metrics']['test_accuracy']:.4f}")