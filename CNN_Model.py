import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import roc_curve, auc 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_data(directory, cell_type):
    position_data = []
    velocity_data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, filename))
            position = df[['X', 'Y', 'Z']].values
            velocity = df[['velocity_x', 'velocity_y', 'velocity_z']].values
            position_data.append(position)
            velocity_data.append(velocity)
            labels.append(cell_type)
    return position_data, velocity_data, labels

# Load PLT and WBC data
soft_position, soft_velocity, soft_labels = load_data('D:/Assessment/PINN/LSTM-Problem/NewTimeSeriesDataForCells/soft_dir', 0)
rigid_position, rigid_velocity, rigid_labels = load_data('D:/Assessment/PINN/LSTM-Problem/NewTimeSeriesDataForCells/hard_dir', 1)

# Combine data and labels
all_position_data = soft_position + rigid_position
all_velocity_data = soft_velocity + rigid_velocity
all_labels = soft_labels + rigid_labels

# Pad sequences to the same length
max_length = max(max(len(seq) for seq in all_position_data), max(len(seq) for seq in all_velocity_data))
padded_position_data = pad_sequences(all_position_data, maxlen=max_length, dtype='float32', padding='post', truncating='post')
padded_velocity_data = pad_sequences(all_velocity_data, maxlen=max_length, dtype='float32', padding='post', truncating='post')

# Normalize the data
position_scaler = StandardScaler()
velocity_scaler = StandardScaler()

normalized_position_data = np.array([position_scaler.fit_transform(seq) for seq in padded_position_data])
normalized_velocity_data = np.array([velocity_scaler.fit_transform(seq) for seq in padded_velocity_data])

# Convert labels to categorical
labels = to_categorical(all_labels)

# Split the data into training and testing sets
X_pos_train, X_pos_test, X_vel_train, X_vel_test, y_train, y_test = train_test_split(
    normalized_position_data, normalized_velocity_data, labels, test_size=0.2, random_state=42)

def create_model(input_shape):
    # Separate inputs for position and velocity
    position_input = Input(shape=input_shape)
    velocity_input = Input(shape=input_shape)

    # Convolutions for position
    x_pos = Conv1D(64, kernel_size=3, activation='relu')(position_input)
    x_pos = MaxPooling1D(2)(x_pos)
    x_pos = Conv1D(64, kernel_size=3, activation='relu')(x_pos)
    x_pos = GlobalAveragePooling1D()(x_pos)

    # Convolutions for velocity
    x_vel = Conv1D(64, kernel_size=3, activation='relu')(velocity_input)
    x_vel = MaxPooling1D(2)(x_vel)
    x_vel = Conv1D(64, kernel_size=3, activation='relu')(x_vel)
    x_vel = GlobalAveragePooling1D()(x_vel)

    # Combine features
    combined = concatenate([x_pos, x_vel])

    # Dense layers
    x = Dense(128, activation='relu')(combined)
    x = Dense(64, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)

    # Create model
    model = Model(inputs=[position_input, velocity_input], outputs=output)
    return model

def create_all_outputs_model(model):
    return Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

input_shape = (max_length, 3)  # 3 features for both position and velocity
model = create_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Custom callback to track test accuracy
class TestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        X_pos_test, X_vel_test, y_test = self.test_data
        test_loss, test_accuracy = self.model.evaluate([X_pos_test, X_vel_test], y_test, verbose=0)
        self.test_accuracies.append(test_accuracy)
        print(f'\nTest accuracy at epoch {epoch+1}: {test_accuracy:.4f}')

# Instantiate the callback
test_accuracy_callback = TestAccuracyCallback((X_pos_test, X_vel_test, y_test))

# Train the model with the custom callback included
cnn_history = model.fit(
    [X_pos_train, X_vel_train], y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[test_accuracy_callback]
)

all_outputs_model = create_all_outputs_model(model)

# Define colors for each class
colors = ['darkorange', 'blue'] 

# Evaluate the model
test_loss, test_accuracy = model.evaluate([X_pos_test, X_vel_test], y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot for Training, Validation, and Test Accuracy
plt.figure(figsize=(7, 6))

plt.plot(cnn_history.history['accuracy'], label='Training Accuracy', color=colors[0], linestyle='--', linewidth=2, marker='o', markersize=5)
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy', color=colors[1], linestyle='--', linewidth=2, marker='s', markersize=5)
plt.plot(test_accuracy_callback.test_accuracies, label='Test Accuracy', color='green', linestyle='--', linewidth=2, marker='^', markersize=5)

plt.title('Accuracy Curve CNN', fontsize=16, fontweight='bold')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(fontsize=12, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

sample_pos = X_pos_test[:1]
sample_vel = X_vel_test[:1]

all_layer_outputs = all_outputs_model.predict([sample_pos, sample_vel])

# Prediction for a sample
def predict_cell_type(trajectory):
    new_position = new_trajectory[:, :3]
    new_velocity = new_trajectory[:, 3:]
    padded_position = pad_sequences([new_position], maxlen=max_length, dtype='float32', padding='post', truncating='post')
    padded_velocity = pad_sequences([new_velocity], maxlen=max_length, dtype='float32', padding='post', truncating='post')
    normalized_position = position_scaler.transform(padded_position[0])
    normalized_velocity = velocity_scaler.transform(padded_velocity[0])
    position_trajectory = normalized_position.reshape(1, max_length, 3)
    velocity_trajectory = normalized_velocity.reshape(1, max_length, 3)
    prediction = model.predict([position_trajectory, velocity_trajectory])
    cell_type = "soft" if prediction[0][0] > prediction[0][1] else "rigid"
    return cell_type, prediction[0][0], prediction[0][1]

example_csv_path = 'D:/Assessment/PINN/LSTM-Problem/NewTimeSeriesDataForCells/hard_dir/output_0.0 (1)_split_1.csv'
new_trajectory = pd.read_csv(example_csv_path)[['X', 'Y', 'Z','velocity_x', 'velocity_y', 'velocity_z']].values
predicted_cell_type_1 = predict_cell_type(new_trajectory)
print(f"Predicted cell type: {predicted_cell_type_1}")

# Apply seaborn style and color palette
sns.set(style="whitegrid")
colors = sns.color_palette("muted")

# Compute ROC curve and ROC area for the model
y_pred = model.predict([X_pos_test, X_vel_test])
fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred[:, 1])
roc_auc = auc(fpr, tpr)

# Plot for ROC curve
plt.figure(figsize=(7, 6))

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})', marker='o', markersize=5)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve (CNN)', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

# Annotate the ROC curve at specific points
for i, j in zip([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8]):
    plt.annotate(f"({i:.1f}, {j:.1f})", (i, j), textcoords="offset points", xytext=(5,5), ha='center', fontsize=10)

plt.tight_layout()
plt.show()
