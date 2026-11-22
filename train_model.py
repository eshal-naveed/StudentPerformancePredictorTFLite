"""
Student Performance Predictor - model training and TFLite conversion.

Trains a small neural network that predicts the probability that a student
passes, from study hours, attendance and previous marks, then converts the
model to TensorFlow Lite (student_model.tflite) for use in the Android app.

Run in Google Colab or locally:  python train_model.py
Author: Eshal Naveed (242201027) - Mobile App Development, BSCS 5A
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# 1. Create a synthetic dataset (2000 students)
np.random.seed(42)
N = 2000
hours = np.random.uniform(0, 10, N)        # study hours per day
attendance = np.random.uniform(0, 100, N)  # attendance percentage
prev = np.random.uniform(0, 100, N)        # previous marks percentage

# Pass/Fail rule: weighted score plus a little random noise
score = (0.35 * hours / 10 + 0.30 * attendance / 100 + 0.35 * prev / 100
         + np.random.normal(0, 0.05, N))
y = (score > 0.5).astype(np.float32)       # 1 = Pass, 0 = Fail

# 2. Scale the inputs to the 0-1 range (the Android app does the same)
X = np.stack([hours / 10, attendance / 100, prev / 100], axis=1).astype(np.float32)

# 3. Split into training (80%) and test (20%) sets
idx = np.random.permutation(N)
split = int(0.8 * N)
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

# 4. Build a small neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')   # output: probability of passing
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Train the model
model.fit(X_train, y_train, epochs=80, batch_size=32,
          validation_data=(X_test, y_test), verbose=1)

# 6. Evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", round(acc * 100, 2), "%")
pred = (model.predict(X_test) > 0.5).astype(int).ravel()
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["Fail", "Pass"]))

# 7. Quick sanity test: [hours/10, attendance/100, previous marks/100]
print(model.predict(np.array([[0.9, 0.95, 0.85], [0.1, 0.3, 0.2]])))

# 8. Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("student_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Model size (KB):", len(tflite_model) / 1024)

# 9. Download the .tflite file (only works inside Google Colab)
try:
    from google.colab import files
    files.download("student_model.tflite")
except ImportError:
    print("Not running in Colab: copy student_model.tflite from this folder.")
