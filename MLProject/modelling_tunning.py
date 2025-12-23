import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import mlflow
import dagshub
import xgboost as xgb

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils import estimator_html_repr
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


# KONEKSI DAGSHUB & MLFLOW
DAGSHUB_USERNAME = "YohanLagi"
DAGSHUB_REPOSITORY = "modelling_analisis_sentimen"

print("Inisialisasi koneksi ke DagsHub...")
dagshub.init(
    repo_owner=DAGSHUB_USERNAME,
    repo_name=DAGSHUB_REPOSITORY,
    mlflow=True
)

mlflow.set_experiment("Sentiment Analysis - XGBoost Hyperparameter Search")

# LOAD DATASET
print("Memuat dataset...")

DATA_PATH = "namadataset_preprocessing/preprocessed_kredivo.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset tidak ditemukan di {DATA_PATH}. "
        "Pastikan file sudah di-commit ke repository."
    )

dataset = pd.read_csv(DATA_PATH)


# LABEL SENTIMEN
def sentiment_mapper(score):
    return "negatif" if score <= 3 else "positif"

dataset["sentiment_category"] = dataset["score"].apply(sentiment_mapper)

# BALANCING DATA (UNDERSAMPLING)
min_samples = dataset["sentiment_category"].value_counts().min()
balanced_df = (
    dataset
    .groupby("sentiment_category")
    .apply(lambda d: d.sample(min_samples, random_state=42))
    .reset_index(drop=True)
)

# ENCODING LABEL
encoder = LabelEncoder()
balanced_df["target"] = encoder.fit_transform(balanced_df["sentiment_category"])

X = balanced_df["text_final"].astype(str).values
y = balanced_df["target"].values

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Dataset siap digunakan | Train: {X_train.shape}, Test: {X_test.shape}")

# HYPERPARAMETER SEARCH
print("Memulai proses hyperparameter tuning...")

search_space = {
    "n_estimators": hp.quniform("n_estimators", 50, 300, 1),
    "max_depth": hp.quniform("max_depth", 3, 10, 1),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
    "subsample": hp.uniform("subsample", 0.5, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
}


def tuning_objective(hyperparams):
    hyperparams["n_estimators"] = int(hyperparams["n_estimators"])
    hyperparams["max_depth"] = int(hyperparams["max_depth"])

    pipeline_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("classifier", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **hyperparams
        ))
    ])

    cv_score = cross_val_score(
        pipeline_model,
        X_train,
        y_train,
        cv=3,
        scoring="accuracy"
    ).mean()

    return {"loss": -cv_score, "status": STATUS_OK}


trials = Trials()
optimal_params = fmin(
    fn=tuning_objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials,
    rstate=np.random.default_rng(42)
)

optimal_params["n_estimators"] = int(optimal_params["n_estimators"])
optimal_params["max_depth"] = int(optimal_params["max_depth"])

print("Parameter terbaik ditemukan:")
print(optimal_params)

# TRAINING MODEL FINAL
with mlflow.start_run(run_name="Final_XGBoost_Model"):

    final_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("classifier", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **optimal_params
        ))
    ])

    final_model.fit(X_train, y_train)

    predictions = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, predictions)

    print(f"Akurasi Akhir: {final_accuracy}")
    print(classification_report(y_test, predictions))

    # Log ke MLflow
    mlflow.log_params(optimal_params)
    mlflow.log_metric("accuracy", final_accuracy)
    mlflow.sklearn.log_model(final_model, "model")

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=encoder.classes_)
    cm_display.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # METRIC JSON
    metrics_summary = {
        "accuracy": final_accuracy,
        "best_params": optimal_params,
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True
        )
    }

    with open("metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    mlflow.log_artifact("metrics_summary.json")

    # ESTIMATOR HTML
    with open("model_structure.html", "w", encoding="utf-8") as f:
        f.write(estimator_html_repr(final_model))

    mlflow.log_artifact("model_structure.html")

print("Training dan logging selesai. Silakan cek dashboard DagsHub.")

