import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score

import pickle
import json
import yaml
import logging

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_pickle(pk_file_name:str):
    try:
        logger.debug("Pickle file retrieved.")
        return pickle.load(open(pk_file_name,'rb'))
    except yaml.YAMLError as e:
        logger.error("Yaml file error occured.")
        raise
    except FileNotFoundError:
        logger.error("Pickle file not found.")
        raise
    except Exception as e:
        print("An unknown error occured.")
        print(e)
        raise

def load_data(train_data_path:str):
    try:
        logger.debug("Retrieved Trained Data.")
        return pd.read_csv(train_data_path)
    except FileNotFoundError:
        logger.error("Train file not found.")
        raise
    except Exception as e:
        print("An unknown error while loading the data.")
        print(e)
        raise

def splitting_data(test_data:pd.DataFrame):
    try:
        X_test = test_data.iloc[:,0:-1].values
        y_test = test_data.iloc[:,-1].values
        logger.debug("Splitted data successfuly")
        return X_test, y_test
    except Exception as e:
        print("Error occured while spitting the data.")
        print(e)
        raise

# Make predictions
def making_predictions(X_test,clf):
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        logger.debug("Predictions Done.")
        return y_pred, y_pred_proba
    except Exception as e:
        print("Error occured while making predictions.")
        print(e)
        raise

# Calculate evaluation metrics
def calcuation(y_test,y_pred,y_pred_proba):
    try:
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) 

        metrics_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'auc' : auc
        }
        logger.debug("Calculations Done Successfully.")
        return metrics_dict
    except Exception as e:
        print("Error occured while calculating.")
        print(e)
        raise

def save_metrics(metrics_dict):
    try:
        with open('metrics.json','w') as file:
            json.dump(metrics_dict,file,indent=4)
        logger.debug("Saved Metrics Successfully.")
    except Exception as e:
        print("Error occured while saving metrics.")
        print(e)
        raise

def main():
    clf = load_pickle('model.pkl')
    test_data = load_data('./data/interim/test_bow.csv')
    X_test, y_test = splitting_data(test_data=test_data)
    y_pred,y_pred_proba = making_predictions(X_test=X_test,clf=clf)
    metrics_dict = calcuation(y_test=y_test,y_pred=y_pred,y_pred_proba=y_pred_proba)
    save_metrics(metrics_dict=metrics_dict)

if __name__ == '__main__':
    main() 