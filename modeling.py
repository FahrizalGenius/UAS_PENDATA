from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_and_predict(model_name, X_train, X_test, y_train, y_test, le):
    if model_name == "SVM":
        model = SVC()
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    elif model_name == "Naive Bayes":
        model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    class_names = [str(cls) for cls in le.classes_]
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=False)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, ax=ax)
    plt.title(f"Confusion Matrix - {model_name}")

    return {"report": report, "cm_plot": fig}
