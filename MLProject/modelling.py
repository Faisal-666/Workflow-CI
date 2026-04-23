import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, precision_score, recall_score, 
    f1_score, ConfusionMatrixDisplay, accuracy_score, confusion_matrix, RocCurveDisplay
)
import mlflow
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument('--train_csv', type=str, default='./diabetes_preprocessing/train.csv')
parser.add_argument('--test_csv', type=str, default='./diabetes_preprocessing/test.csv')
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

train_df = pd.read_csv(args.train_csv)
test_df = pd.read_csv(args.test_csv)

x_train = train_df.drop('Outcome', axis=1)
y_train = train_df['Outcome']

x_test = test_df.drop('Outcome', axis=1)
y_test = test_df['Outcome']

input_example = x_train.iloc[0:3]

with mlflow.start_run():
    model = LogisticRegression(random_state=args.random_state, max_iter=args.max_iter)
    model.fit(x_train, y_train)

    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path='model', 
        input_example=input_example,
        registered_model_name='Diabetes_Logistic_Regression'
    )

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_proba))
    mlflow.log_metric('precision', precision_score(y_test, y_pred))
    mlflow.log_metric('recall', recall_score(y_test, y_pred))
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred))
    mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))

    cm_display = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    mlflow.log_artifact('confusion_matrix.png')

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    mlflow.log_metric('specificity', specificity)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_dict(report_dict, 'classification_report.json')

    metric_info = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'specificity': specificity
    }

    with open('metric_info.json', 'w') as f:
        json.dump(metric_info, f, indent=4)

    mlflow.log_artifact('metric_info.json')

    coef_df = pd.DataFrame({
        'feature': x_train.columns,
        'coefficient': model.coef_[0]
    })
    coef_df.to_csv('feature_coefficients.csv', index=False)
    mlflow.log_artifact('feature_coefficients.csv') 

    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
    plt.title('ROC Curve')
    plt.savefig('roc_curve.png')
    plt.close(fig)
    mlflow.log_artifact('roc_curve.png')