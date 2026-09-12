import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(16, input_dim=4, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train, 
    epochs=25, 
    batch_size=5, 
    validation_split=0.1, 
    verbose=0
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

user_input = np.array([[5.1, 3.5, 1.4, 0.2]]) 
user_input_scaled = scaler.transform(user_input)

prediction = model.predict(user_input_scaled, verbose=0)
predicted_class_index = np.argmax(prediction)
predicted_class_name = iris.target_names[predicted_class_index]

print("EXPERIMENT 2 – OUTPUT\n")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}\n")
print(f"Predicted Iris Species: {predicted_class_name}\n")

plt.figure(figsize=(7, 5))
plt.plot(history.history['accuracy'], marker='o', color='black', linewidth=1, markersize=3)
plt.title('MLP Training Accuracy on Iris Dataset', fontsize=11, fontweight='bold')
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.ylim(0.0, 1.0)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()