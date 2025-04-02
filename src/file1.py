from ast import Tuple
import logging
import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

reports_dir = './reports'
os.makedirs(reports_dir, exist_ok=True)
#Ensure if log dir is existing
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging Config
logger = logging.getLogger('MLFlow')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'MLFlow.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

mlflow.set_tracking_uri('https://5000-01jqpxkt9h0g5hx4cwe8bcwak3.cloudspaces.litng.ai')
mlflow.set_experiment('YT-MLOPS-Exp1')

def load_dataset(test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        wine = load_wine()
        x = wine.data
        y = wine.target
        logger.debug(f'Wine Dataset loaded sucessfully')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)
        logger.debug(f'Wine Dataset Splitted sucessfully')
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logger.error(f'Error Raised while loading the wine dataset as: {e}')
        raise

def model_training_pred(max_depth: int, n_estimators: int, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    try:
        logger.debug(f'Model Training Initiated')
        rf = RandomForestClassifier(max_depth = max_depth, n_estimators=n_estimators, random_state=42)
        rf.fit(x_train, y_train)
        logger.debug(f'Model Training done')
        y_pred = rf.predict(x_test)
        return y_pred
    except Exception as e:
        logger.error(f'Unwanted error Raised as: {e}')
        raise

def metrics_logging(max_depth: int, n_estimators: int,  y_test: np.ndarray, y_pred: np.ndarray) -> None:
    try:
        wine = load_wine()
        accuracy = accuracy_score(y_test, y_pred)
        logger.debug(f'Accuraacy of model calculated: {accuracy}')

        mlflow.log_metric('Accuracy', accuracy)
        mlflow.log_metric('max_depth', max_depth)
        mlflow.log_metric('n_estimators', n_estimators)
        logger.debug(f'Metrics logged at mlFlow')

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig("./reports/Confusion-matrix.png")
        logger.debug(f'cm saved at path: ./reports/Confusion-matrix.png')

        mlflow.log_artifact("./reports/Confusion-matrix.png")
        mlflow.log_artifact(__file__)

        mlflow.set_tags({"Author": 'Nan ji Bhai', "Project": "Wine Classification"})
        return
    except FileNotFoundError as e:
        logger.error(f'File not found at location as: {e}')
        raise
    except Exception as e:
        logger.error(f'Unwanted Exception raised as: {e}')
        raise

def main():
    with mlflow.start_run():
        max_depth = 15
        n_estimators = 20
        x_train, x_test, y_train, y_test = load_dataset(0.2)
        y_pred = model_training_pred(max_depth, n_estimators, x_train, y_train, x_test)
        
        metrics_logging(max_depth, n_estimators, y_test, y_pred)

if __name__ == '__main__':
    main()

