from sklearn.ensemble import RandomForestClassifier


def create_model():  # Create a machine learning model for anomaly detection.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    return model


def train_model(model, x_train, y_train):  # Train the model on training data.

    model.fit(x_train, y_train)
    return model


def predict(model, X):  # Make predictions with the trained model.

    return model.predict(X)
