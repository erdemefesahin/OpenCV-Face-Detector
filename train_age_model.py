"""
Fast & Lightweight Age Classification Training Script
======================================================
Trains a small CNN model for age classification that trains in 5-10 minutes on CPU.

Dataset Structure:
    dataset/
        0-2/
        4-6/
        8-12/
        15-20/
        25-32/
        38-43/
        48-53/
        60-100/

Outputs:
    - age_model.h5 (trained model)
    - age_labels.json (class labels)
    - training_plots.png (accuracy/loss graphs)
    - confusion_matrix.png (confusion matrix)

Author: Fast CNN Trainer
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Configuration
IMG_SIZE = 80  # Small size for fast training
BATCH_SIZE = 32
EPOCHS = 12
VALIDATION_SPLIT = 0.2
DATASET_PATH = 'dataset'
MODEL_PATH = 'age_model.h5'
LABELS_PATH = 'age_labels.json'
PLOTS_PATH = 'training_plots.png'
CONFUSION_PATH = 'confusion_matrix.png'


def build_fast_cnn(input_shape, num_classes):
    """
    Build a lightweight CNN with ~200K-500K parameters.
    Fast to train, small memory footprint.
    """
    model = models.Sequential([
        # Block 1: 32 filters
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2: 64 filters
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3: 128 filters
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_data_generators():
    """Create training and validation data generators with augmentation."""
    
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT
    )
    
    # Validation generator (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )
    
    return train_datagen, val_datagen


def plot_training_history(history):
    """Plot and save training accuracy and loss curves."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2, color='#2ecc71')
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#e74c3c')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training', linewidth=2, color='#2ecc71')
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2, color='#e74c3c')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_PATH, dpi=150, bbox_inches='tight')
    print(f"✅ Saved training plots to: {PLOTS_PATH}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Generate and save confusion matrix."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Age Classification', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Age Group', fontsize=12)
    plt.xlabel('Predicted Age Group', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=150, bbox_inches='tight')
    print(f"✅ Saved confusion matrix to: {CONFUSION_PATH}")
    plt.close()


def train_model():
    """Main training function."""
    
    print("="*70)
    print("🚀 FAST & LIGHTWEIGHT AGE CLASSIFICATION TRAINING")
    print("="*70)
    print()
    
    # Check dataset
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset folder '{DATASET_PATH}' not found!")
        print(f"\nExpected structure:")
        print(f"  {DATASET_PATH}/")
        print(f"    0-2/")
        print(f"    4-6/")
        print(f"    8-12/")
        print(f"    ...")
        return
    
    # Get age groups
    age_groups = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    num_classes = len(age_groups)
    
    if num_classes == 0:
        print(f"❌ Error: No age group folders found in '{DATASET_PATH}'")
        return
    
    print(f"📊 Dataset Information:")
    print(f"   Path: {DATASET_PATH}")
    print(f"   Age Groups: {num_classes}")
    print(f"   Classes: {age_groups}")
    print()
    
    # Create data generators
    print("📁 Setting up data generators...")
    train_datagen, val_datagen = create_data_generators()
    
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Save class labels
    class_indices = train_generator.class_indices
    label_mapping = {v: k for k, v in class_indices.items()}
    with open(LABELS_PATH, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"✅ Saved class labels to: {LABELS_PATH}")
    
    print()
    print(f"📈 Training samples: {train_generator.samples}")
    print(f"📉 Validation samples: {validation_generator.samples}")
    print()
    
    # Build model
    print("🏗️  Building lightweight CNN model...")
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    model = build_fast_cnn(input_shape, num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    print()
    print("📋 Model Architecture:")
    model.summary()
    
    total_params = model.count_params()
    print()
    print(f"⚡ Total Parameters: {total_params:,} (~{total_params/1000:.0f}K)")
    print()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    
    # Training
    print("="*70)
    print("🎯 STARTING TRAINING")
    print("="*70)
    print(f"   Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Early Stopping: Patience=4")
    print()
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate on validation set
    print()
    print("="*70)
    print("📊 FINAL EVALUATION")
    print("="*70)
    
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_accuracy*100:.2f}%")
    print()
    
    # Generate predictions for confusion matrix
    print("🔮 Generating predictions for confusion matrix...")
    validation_generator.reset()
    y_pred_probs = model.predict(validation_generator, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = validation_generator.classes
    
    # Classification report
    print()
    print("📋 Classification Report:")
    print()
    print(classification_report(y_true, y_pred, target_names=age_groups))
    
    # Save model
    print("💾 Saving model...")
    model.save(MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"   Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    print()
    
    # Generate plots
    print("📊 Generating visualizations...")
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred, age_groups)
    print()
    
    # Summary
    print("="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print()
    print("📦 Generated Files:")
    print(f"   ✅ {MODEL_PATH} - Trained model")
    print(f"   ✅ {LABELS_PATH} - Class labels")
    print(f"   ✅ {PLOTS_PATH} - Training graphs")
    print(f"   ✅ {CONFUSION_PATH} - Confusion matrix")
    print()
    print("🚀 You can now use the model with predict_age.py!")
    print("="*70)


if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train
    train_model()
