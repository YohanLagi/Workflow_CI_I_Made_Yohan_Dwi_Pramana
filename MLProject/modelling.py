import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# MLflow Configuration
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Sentiment Analysis - Kredivo")
mlflow.sklearn.autolog()

# Load Data
df = pd.read_csv("preprocessed_kredivo.csv")

# Validasi kolom wajib
required_cols = {"text_final", "score"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Kolom hilang: {missing}")

# Label Sentimen
df["sentiment_category"] = df["score"].apply(
    lambda x: "negatif" if x <= 3 else "positif"
)

# Undersampling
min_size = df["sentiment_category"].value_counts().min()
df_balanced = (
    df.groupby("sentiment_category", group_keys=False)
      .apply(lambda d: d.sample(min_size, random_state=42))
      .reset_index(drop=True)
)

# Encoding & Split
le = LabelEncoder()
df_balanced["label"] = le.fit_transform(df_balanced["sentiment_category"])

X = df_balanced["text_final"].astype(str).values
y = df_balanced["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TRAINING 
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("xgb", xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
