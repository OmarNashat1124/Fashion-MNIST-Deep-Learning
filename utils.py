import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

MODEL_PATH = "model/fashion_mnist_model.keras"

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x_train, x_test):
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train_flat = x_train.reshape(len(x_train), 28 * 28)
    x_test_flat = x_test.reshape(len(x_test), 28 * 28)
    return x_train_flat, x_test_flat

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(100, input_shape=(784,), activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=10):
    history = model.fit(x_train, y_train, epochs=epochs)
    return history

def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"✅ Test accuracy: {test_acc:.4f}")
    return test_loss, test_acc

def save_model(model, model_path=MODEL_PATH):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")

def load_saved_model(model_path=MODEL_PATH):
    """Load previously trained model."""
    model = keras.models.load_model(model_path)
    print(f"📂 Loaded model from {model_path}")
    return model

def predict(model, x_input):
    """Predict for single or batch of images."""
    if x_input.ndim == 1:
        x_input = np.expand_dims(x_input, axis=0)
    preds = model.predict(x_input)
    predicted_class = np.argmax(preds, axis=1)
    return predicted_class, preds


def predict(model, x_test):
    """Predict labels for test data."""
    predictions = model.predict(x_test)
    return predictions

def predict_image(model, img_path):
    img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    img_arr = np.array(img).astype("float32") / 255.0
    img_flat = img_arr.reshape(1, 28 * 28)
    pred_class, _ = predict(model, img_flat)
    return pred_class
