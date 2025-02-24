import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

import pickle
import yaml
import logging

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# getting data from 'params.yaml' 
def get_yaml(yaml_file:str):
    try:
        params = yaml.safe_load(open(yaml_file,'r'))['model_building']
        logger.debug("Yaml file reterieved.")
        return params
    except yaml.YAMLError as e:
        logger.error('yaml error')
        raise
    except FileNotFoundError:
        logger.error('Yaml file not found.')
        raise
    except Exception as e:
        print("Unknown Error Occured.")
        print(e)
        raise

# fetching the data
def get_data(train_data_path:str):
    try:
        train_data = pd.read_csv(train_data_path)
        logger.debug("Retrieved training data.")
        return train_data
    except FileNotFoundError:
        logger.error("Training data not found.")
        raise
    except Exception as e:
        print("Unknonwn error occured.")
        print(e)
        raise

def split_data(train_data):
    try:
        X_train = train_data.iloc[:,0:-1].values
        y_train = train_data.iloc[:,-1].values
        logger.debug("Splitted the data successfully.")
        return X_train,y_train
    except IndexError:
        logger.error("Index error occured.")
        raise
    except Exception as e:
        logger.error("Unknown error occured.")
        raise

# Define and train the XGBoost model
def build_model(model,X_train,y_train):
    try:
        clf = GradientBoostingClassifier(n_estimators=model['n_estimators'],learning_rate=model['learning_rate'])
        clf.fit(X_train, y_train)
        logger.debug('Made & Trained Model.')
        # save the file in pickle
        pickle.dump(clf,open('model.pkl','wb'))
        logger.debug('Made the pickle file.')
    except Exception as e:
        print("Error occured while building model.")
        print(e)
        raise

def main():
    model = get_yaml('params.yaml')
    train_data = get_data('./data/interim/train_tfidf.csv')
    X_train,y_train = split_data(train_data=train_data)
    build_model(model,X_train=X_train,y_train=y_train)

if __name__ == '__main__':
    main()