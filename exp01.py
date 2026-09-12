import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
(X_train, y_train), (X_test,y_test) = mnist.load_data()
X_train = X_train.reshape(-1,784)/255.0
X_test = X_test.reshape(-1,784)/255.0
encoder=OneHotEncoder(sparse_output=False)
y_train_encoded=encoder.fit_transform(y_train.reshape(-1,1))
input_size=784
hidden_size=64
output_size=10
np.random.seed(42)
W1=np.random.randn(input_size,hidden_size)*0.01
b1=np.zeros((1,hidden_size))
W2=np.random.randn(hidden_size,output_size)*0.01
b2=np.zeros((1,output_size))
sigmoid=lambda x: 1/(1 + np.exp(-np.clip(x,-500,500)))
sigmoid_deriv=lambda a: a*(1-a)
loss_fn=lambda y, y_hat: -np.sum(y*np.log(y_hat + 1e-8))
epochs=10
lr=0.01
losses=[]
print("Starting Training...")
for epoch in range(epochs):
    total_loss=0.0
    for i in range(X_train.shape[0]):
        x = X_train[i:i+1]            
        y = y_train_encoded[i:i+1]
        z1 = x @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        loss = loss_fn(y, a2)
        total_loss += loss
        dz2 = a2 - y                  
        dW2 = a1.T @ dz2
        db2 = dz2
        dz1 = (dz2 @ W2.T) * sigmoid_deriv(a1) 
        dW1 = x.T @ dz1
        db1 = dz1
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
    avg_loss = total_loss / X_train.shape[0]
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), losses, marker='o', color='b', linewidth=2)
plt.title("Training Loss Progression Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.show()
def predict(img_vector):
    img = img_vector.reshape(1, 784)
    a1 = sigmoid(img @ W1 + b1)
    a2 = sigmoid(a1 @ W2 + b2)
    return np.argmax(a2)
sample_idx = 100
sample_img = X_train[sample_idx]
predicted_label = predict(sample_img)

plt.figure(figsize=(4, 4))
plt.imshow(sample_img.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_label}", fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()