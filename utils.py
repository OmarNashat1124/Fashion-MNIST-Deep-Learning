import tensorflow as tf
from tensorflow import keras

def load_data():
    """Load and return Fashion MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x_train, x_test):
    """Normalize and flatten image data."""
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train_flat = x_train.reshape(len(x_train), 28 * 28)
    x_test_flat = x_test.reshape(len(x_test), 28 * 28)
    return x_train_flat, x_test_flat

def build_model():
    """Build and compile a simple dense neural network."""
    model = keras.Sequential([
        keras.layers.Dense(100, input_shape=(784,), activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=10):
    """Train the model."""
    history = model.fit(x_train, y_train, epochs=epochs)
    return history

def evaluate_model(model, x_test, y_test):
    """Evaluate model accuracy on test data."""
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    return test_loss, test_acc

def predict(model, x_test):
    """Predict labels for test data."""
    predictions = model.predict(x_test)
    return predictions
