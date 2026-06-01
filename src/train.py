from generate_signals import create_dataset
from features import extract_features
from sklearn.model_selection import train_test_split
from model import create_model
from sklearn.metrics import classification_report


X, y = create_dataset(2000)
X = extract_features(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


model = create_model()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))