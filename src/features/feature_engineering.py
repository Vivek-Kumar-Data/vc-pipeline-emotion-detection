import pandas as pd

import os
import yaml
import logging

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_yaml(param_path:str):
    try : 
        max_features = yaml.safe_load(open(param_path,'r'))['feature_engineering']['max_features']
        logger.debug("max_features value reterived successfully.")
        return max_features
    except yaml.YAMLError as e:
        logger.error('yaml error')
        raise
    except FileNotFoundError:
        logger.error('Params file Not found.')
        raise

def get_data(train_path:str,test_path:str):

    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug("Read train and test data")
        return train_data, test_data
    
    except FileNotFoundError:
        logger.error('Train/Test data Not Found')
        raise

    except yaml.YAMLError as e:
        logger.error('Yaml Error !!!')
        print(e)
        raise

    except Exception as e:
        logger.error("Some unknown error occured at get data.")

def preprocessing(train_data,test_data,max_features):
    try:
        # filling the null values 
        train_data.fillna('',inplace=True)
        test_data.fillna('',inplace=True)
        logger.debug("filled missing values")
        # splitting the train data
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        # splitting the test data
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        logger.debug('got values of x/y trian&test.')
        # Apply Bag of Words (CountVectorizer)
        vectorizer = TfidfVectorizer(max_features=max_features)
        # Fit the vectorizer on the training data and transform it
        X_train_bow = vectorizer.fit_transform(X_train)
        # Transform the test data using the same vectorizer
        X_test_bow = vectorizer.transform(X_test)
        logger.debug("Bag of Words applied successsfully.")
        # converting the final result to array
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        logger.debug("got the train_df & test_df data.")

        return train_df, test_df
    
    except FileNotFoundError:
        logger.error("File not found")
        raise

    except Exception as e:
        logger.error("Some unknown error occured.")
        print(e)
        raise

def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame,data_path:str) -> None:
    try:
        # store the data in data/
        # first we will create a path
        data_path = os.path.join(data_path,'interim') 
        logger.debug("data path reterieved successfully.")
        # now we will make a new directory
        os.makedirs(data_path)  
        logger.debug("directory made successfully")
        # yaha par hamara data wala folder ban jayega our raw wala folder ban jayega.
        # now we will save the file to our new directory.
        train_data.to_csv(os.path.join(data_path,'train_tfidf.csv'))
        test_data.to_csv(os.path.join(data_path,'test_tfidf.csv'))
        logger.debug("output files successfully created to path : ",data_path)

    except Exception as e:
        print("Some error occured")
        print(e)
        raise

def main():
    # getting max features from the yaml file.
    max_features = load_yaml(param_path='params.yaml')
    # getting the data path
    train_data_path = './data/processed/train_processed.csv'
    test_data_path = './data/processed/test_processed.csv'
    # getting the data from the desired location
    train_data, test_data = get_data(train_path=train_data_path,test_path=test_data_path)
    # preprocessing
    train_data,test_data = preprocessing(train_data=train_data,test_data=test_data,max_features=max_features)
    # saving the data
    save_data(train_data=train_data,test_data=test_data,data_path='data')

if __name__ == '__main__':
    main()