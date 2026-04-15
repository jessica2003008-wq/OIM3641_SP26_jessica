"""
Summary:
In this project, I compared a low-code machine learning workflow using PyCaret
with a manual workflow using scikit-learn on the Adult Income classification dataset.
PyCaret was more efficient because it automated preprocessing, model comparison,
and evaluation in only a few lines of code. This saved time and made it easy to
identify strong baseline models quickly. For a data engineer, this low-code workflow
is useful when speed and experimentation matter.

The manual scikit-learn workflow required more work because I had to explicitly
split the data, identify categorical and numerical columns, build preprocessing
pipelines, encode text features, scale numerical features, and train the model.
Although this process took longer, it gave me more transparency and control over
how the data moved through the pipeline. This makes scikit-learn a better choice
when customization or deeper understanding of the pipeline is important.

The results may differ slightly between PyCaret and scikit-learn because of
differences in random state behavior, preprocessing defaults, cross-validation,
and hyperparameter settings. Even when the same algorithm is used, small
implementation differences can produce slightly different performance metrics.
Overall, PyCaret was more efficient for fast experimentation, while scikit-learn
provided more flexibility and interpretability.
"""

import pandas as pd
from sklearn.datasets import fetch_openml

# 1. Load dataset
adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame.copy()

print("Columns:")
print(df.columns.tolist())
print("\nShape:", df.shape)

# Make target column easier to reference
df["income"] = df["class"]
df = df.drop(columns=["class"])

print("\nTarget counts:")
print(df["income"].value_counts())

# 2. PyCaret workflow
from pycaret.classification import setup, compare_models, pull, plot_model, save_model

clf_setup = setup(
    data=df,
    target="income",
    session_id=42,
    train_size=0.8,
    normalize=True,
    verbose=False
)

top3 = compare_models(n_select=3)
comparison_results = pull()

print("\nPyCaret Model Comparison Table:")
print(comparison_results)

comparison_results.to_csv("pycaret_model_comparison.csv", index=False)

best_model = top3[0]

plot_model(best_model, plot="confusion_matrix", save=True)

save_model(best_model, "best_pipeline")

# 3. Manual scikit-learn workflow
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

X = df.drop(columns=["income"])
y = df["income"]

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

manual_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

manual_pipeline.fit(X_train, y_train)
y_pred = manual_pipeline.predict(X_test)

print("\nManual Scikit-Learn Classification Report:")
print(classification_report(y_test, y_pred))