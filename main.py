from utils import load_data, preprocess_data, build_model, train_model, evaluate_model, predict

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train_flat, x_test_flat = preprocess_data(x_train, x_test)
    model = build_model()
    train_model(model, x_train_flat, y_train, epochs=10)
    evaluate_model(model, x_test_flat, y_test)
    preds = predict(model, x_test_flat)
    print("Prediction sample:", preds[0])

if __name__ == "__main__":
    main()
