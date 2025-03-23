import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import librosa

def load_and_preprocess_audio(file_path, sr=16000, duration=3):
    try:
     
        audio, sr = librosa.load(file_path, sr=sr, duration=duration)
        target_length = sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
  
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)

        rmse = librosa.feature.rms(y=audio)
      
        features = np.concatenate([mfcc, spectral_centroid, rmse])
        
        return features.T
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def prepare_dataset(base_path):
    
    X = []
    y = []
    
    if not os.path.exists(base_path):
        raise ValueError(f"Base path does not exist: {base_path}")
    
    
    dysarthric_folders = ['dysarthria_female', 'dysarthria_male']
    non_dysarthric_folders = ['non-dysarthria_female', 'non-dysarthria_male']
    
    print(f"Available folders in {base_path}:")
    print(os.listdir(base_path))
    
    for folder in dysarthric_folders + non_dysarthric_folders:
        folder_path = os.path.join(base_path, folder)
        label = 1 if 'dysarthria_' in folder else 0
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
            
        print(f"\nProcessing {folder}:")
        files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.WAV'))]
        print(f"Found {len(files)} WAV files")
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            print(f"Processing: {file}", end='\r')
            features = load_and_preprocess_audio(file_path)
            if features is not None:
                X.append(features)
                y.append(label)
        
        print(f"\nSuccessfully processed {len([y_ for y_ in y if y_ == label])} files from {folder}")
    
    if len(X) == 0:
        raise ValueError("No audio files were successfully processed!")
        
    return np.array(X), np.array(y)

def create_model(input_shape):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_learning_curves(history):
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(20,8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('plots/loss_curves.png')
    plt.close()
    plt.figure(figsize=(20,8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('plots/accuracy_curves.png')
    plt.close()

def plot_roc_curve(y_test, y_pred):
    plt.figure(figsize=(20,8))
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label=f"CNN Model, AUC={auc:.2f}", lw=2)
    plt.plot([0, 1], [0, 1], color="orange", lw=2, linestyle="--")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc=4)
    plt.savefig('plots/roc_curve.png')
    plt.close()

def plot_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(10,8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, cmap='viridis', annot=True, fmt='.3g',
                xticklabels=['Non Dysarthria','Dysarthria'],
                yticklabels=['Non Dysarthria','Dysarthria'])
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

def main():
    
    base_path = r"E:/Projects/Hackathon Projects/JivaLabs/Datasets/Dysarthia ds"
    
    try:

        print("Loading and preprocessing data...")
        X, y = prepare_dataset(base_path)
        
    
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(y)}")
        print(f"Dysarthric samples: {np.sum(y)}")
        print(f"Non-dysarthric samples: {len(y) - np.sum(y)}")
        print(f"Feature shape: {X.shape}")
    
        scaler = StandardScaler()
        X_reshaped = X.reshape(X.shape[0], -1)
        X_normalized = scaler.fit_transform(X_reshaped)
        X = X_normalized.reshape(X.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
       
        print("\nCreating model...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_model(input_shape)
   
        model_name = "dysarthria_model.keras"
        checkpoint = ModelCheckpoint(
            model_name,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=5,
            verbose=1,
            restore_best_weights=True
        )


        print("\nTraining model...")
        history = model.fit(
            X_train, 
            y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint, early_stopping]
        )

        print("\nGenerating predictions...")
        y_pred = model.predict(X_test)

      
        print("\nPlotting results...")
        plot_learning_curves(history)
        plot_roc_curve(y_test, y_pred)
        
        y_pred_binary = (y_pred >= 0.5).astype(int)
        

        plot_confusion_matrix(y_test, y_pred_binary)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_binary))
        print(f"\nRecall Score: {recall_score(y_test, y_pred_binary):.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()