# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from scipy.stats import skew, kurtosis

# Function to load and preprocess data
def load_data(directory, cell_type):
    data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, filename))
            trajectory = df[['X', 'Y', 'Z', 'velocity_x', 'velocity_y', 'velocity_z']].values
            data.append(trajectory)
            labels.append(cell_type)
    return data, labels

# Load soft and rigid data
soft_data, soft_labels = load_data('D:/Assessment/PINN/LSTM-Problem/NewTimeSeriesDataForCells/soft_dir', 0)
rigid_data, rigid_labels = load_data('D:/Assessment/PINN/LSTM-Problem/NewTimeSeriesDataForCells/hard_dir', 1)

# Combine data and labels
all_data = soft_data + rigid_data
all_labels = soft_labels + rigid_labels

# Pad sequences to the same length
max_length = max(len(seq) for seq in all_data)
padded_data = pad_sequences(all_data, maxlen=max_length, dtype='float32', padding='post', truncating='post')

# Normalize the data
scaler = StandardScaler()
normalized_data = np.array([scaler.fit_transform(seq) for seq in padded_data])

# Convert labels to categorical
labels = to_categorical(all_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=42, stratify=all_labels)

# Define the GRU model for classification
model = Sequential([
    Bidirectional(GRU(128, return_sequences=True), input_shape=(max_length, 6)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(GRU(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(GRU(32)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model with a different learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping and learning rate reduction on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Custom callback to track test accuracy
class TestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.rnn_test_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        X_test, y_test = self.test_data
        test_loss, rnn_test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        self.rnn_test_accuracies.append(rnn_test_accuracy)
        print(f'\nTest accuracy at epoch {epoch+1}: {rnn_test_accuracy:.4f}')

# Instantiate the callback
rnn_test_accuracy_callback = TestAccuracyCallback((X_test, y_test))

# Train the model with a smaller batch size and more epochs
rnn_history = model.fit(X_train, y_train, epochs=120, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr, rnn_test_accuracy_callback])

# Evaluate the model
test_loss, rnn_test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {rnn_test_accuracy:.4f}")

# Apply seaborn style and color palette
sns.set(style="whitegrid")
colors = sns.color_palette("muted")

# Plot for Training and Validation Loss
plt.figure(figsize=(7, 6))

plt.plot(rnn_history.history['loss'], label='Training Loss(RNN)', color=colors[0], linestyle='dotted', linewidth=2, marker='o', markersize=5)
plt.plot(rnn_history.history['val_loss'], label='Validation Loss(RNN)', color=colors[1], linestyle='dotted', linewidth=2, marker='s', markersize=5)
plt.title('Learning Curve RNN', fontsize=16, fontweight='bold')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Plot for Training and Validation Accuracy
plt.figure(figsize=(7, 6))

plt.plot(rnn_history.history['accuracy'], label='Training Accuracy', color=colors[0], linestyle='dotted', linewidth=2, marker='o', markersize=5)
plt.plot(rnn_history.history['val_accuracy'], label='Validation Accuracy', color=colors[1], linestyle='dotted', linewidth=2, marker='s', markersize=5)
plt.plot(rnn_test_accuracy_callback.rnn_test_accuracies, label='Test Accuracy', color='green', linestyle='dotted', linewidth=2, marker='^', markersize=5)

plt.title('Accuracy Curve RNN', fontsize=16, fontweight='bold')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(fontsize=12, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Predict probabilities for the test set
y_pred_prob = model.predict(X_test)

# Compute ROC curve and ROC area for each class 
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Define colors for each class
colors = ['darkorange', 'blue']  # Different colors for each class

# Plot ROC curve
plt.figure(figsize=(8, 6))
for i in range(2):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'{"CNN" if i == 0 else "RNN"} (area = {roc_auc[i]:.2f})', marker='o', markersize=5)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

# Annotate the ROC curve at specific points
for i, j in zip([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8]):
    plt.annotate(f"({i:.1f}, {j:.1f})", (i, j), textcoords="offset points", xytext=(5,5), ha='center', fontsize=10)
plt.show()

# Predict classes for the test set
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Soft', 'Rigid'], yticklabels=['Soft', 'Rigid'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Soft', 'Rigid']))

# Function to predict cell type
def predict_cell_type(trajectory):
    padded_trajectory = pad_sequences([trajectory], maxlen=max_length, dtype='float32', padding='post', truncating='post')
    normalized_trajectory = scaler.transform(padded_trajectory[0])
    normalized_trajectory = normalized_trajectory.reshape(1, max_length, 6)
    prediction = model.predict(normalized_trajectory)
    cell_type = "soft" if prediction[0][0] > prediction[0][1] else "rigid"
    return cell_type, prediction[0][0], prediction[0][1]

# Example usage with a CSV file
example_csv_path = 'D:/Assessment/PINN/LSTM-Problem/NewTimeSeriesDataForCells/hard_long_channel_orginal/run1/output_0.0.csv'
new_trajectory = pd.read_csv(example_csv_path)[['X', 'Y', 'Z', 'velocity_x', 'velocity_y', 'velocity_z']].values
predicted_cell_type = predict_cell_type(new_trajectory)
print(f"Predicted cell type: {predicted_cell_type}")

# Example usage with a CSV file
example_csv_path = 'D:/Assessment/PINN/LSTM-Problem/NewTimeSeriesDataForCells/soft_dir/output_0.0 (1)_split_1.csv'
new_trajectory = pd.read_csv(example_csv_path)[['X', 'Y', 'Z', 'velocity_x', 'velocity_y', 'velocity_z']].values
predicted_cell_type = predict_cell_type(new_trajectory)
print(f"Predicted cell type: {predicted_cell_type}")

# Split data into soft and rigid for visualization
df_soft = pd.DataFrame(np.concatenate(soft_data), columns=['X', 'Y', 'Z', 'velocity_x', 'velocity_y', 'velocity_z'])
df_rigid = pd.DataFrame(np.concatenate(rigid_data), columns=['X', 'Y', 'Z', 'velocity_x', 'velocity_y', 'velocity_z'])

# Visualize the distribution of each feature for soft and rigid data
plt.figure(figsize=(12, 8))
for i, col in enumerate(df_soft.columns):
    plt.subplot(2, 3, i+1)
    sns.histplot(df_soft[col], kde=True, color='blue', label='Soft', stat="density", element="step")
    sns.histplot(df_rigid[col], kde=True, color='red', label='Rigid', stat="density", element="step")
    plt.title(col)
    plt.legend()
plt.tight_layout()
plt.show()

# Print model summary
model.summary()

# Statistical analysis of each feature
stats = pd.DataFrame(index=df_soft.columns)
stats['Soft Mean'] = df_soft.mean()
stats['Soft Median'] = df_soft.median()
stats['Soft Std'] = df_soft.std()
stats['Soft Skewness'] = df_soft.apply(skew)
stats['Soft Kurtosis'] = df_soft.apply(kurtosis)

stats['Rigid Mean'] = df_rigid.mean()
stats['Rigid Median'] = df_rigid.median()
stats['Rigid Std'] = df_rigid.std()
stats['Rigid Skewness'] = df_rigid.apply(skew)
stats['Rigid Kurtosis'] = df_rigid.apply(kurtosis)

print("Statistical Analysis:")
print(stats)