import numpy as np
import pandas as pd

import os
import logging
import yaml
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# fetching the data from data/raw
def get_data(train_data_path:str,test_data_path:str):
    try:
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        logger.debug("reterived data successfully.")
        return train_data, test_data
    except yaml.YAMLError as e:
        logger.error('yaml error')
        raise
    except FileNotFoundError:
        logger.error("Failed to get data, file not found!!!")
        raise
    except Exception as e:
        print("Some unknown error occured.")
        print(e)
        raise

# transfomring the data
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    # making the class of WrodNetLemmatizer
    lemmatizer= WordNetLemmatizer()
    # splitting the data.
    text = text.split()
    #transforming the data.
    text=[lemmatizer.lemmatize(y) for y in text]
    # returning the data.
    return " " .join(text)

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        Text=[i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    except SyntaxError:
        logger.error('Syntax error with the text.')
        raise
    except Exception as e:
        print("Some error occured.")
        print(e)
        raise 

def removing_numbers(text):

    try:
        text=''.join([i for i in text if not i.isdigit()])
        return text
    
    except SyntaxError:
        logger.error('Syntax error with the text.')
        raise

    except Exception as e:
        print("Some error occured")
        print(e)
        raise 

def lower_case(text):
        
    try:
        text = text.split()
        text=[y.lower() for y in text]
        return " " .join(text)
    
    except SyntaxError:
        logger.error('Syntax error with the text.')
        raise

    except Exception as e:
        print("Some error occured")
        print(e)
        raise 

def removing_punctuations(text):
    try:
        ## Remove punctuations
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )
        ## remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
        return text.strip()
    except SyntaxError:
        logger.error('Syntax error with the text.')
        raise
    except Exception as e:
        print("Some error occured")
        print(e)
        raise 

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except SyntaxError:
        logger.error('Syntax error with the text.')
        raise
    except Exception as e:
        print("Some error occured")
        print(e)
        raise 

def remove_small_sentences(df):
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
        logger.debug("Removed small sentences.")
    except SyntaxError:
        logger.error('Syntax error with the text.')
        raise
    except Exception as e:
        print("Some error occured")
        print(e)
        raise 

def normalize_text(df):
    try:
        df.content=df.content.apply(lambda content : lower_case(content))
        logger.debug("converted text to lower case.")
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        logger.debug("stop words removed.")
        df.content=df.content.apply(lambda content : removing_numbers(content))
        logger.debug("removed numbers successfully.")
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        logger.debug("removed punctutation successfully.")
        df.content=df.content.apply(lambda content : removing_urls(content))
        logger.debug("Removed urls successfully.")
        df.content=df.content.apply(lambda content : lemmatization(content))
        logger.debug("Lemmatization done.")
        logger.debug('Normalized text successfully.')
        return df

    except Exception as e:
        print("Some error occured")
        print(e)
        raise 

def get_final(train_data,test_data):
    try:
        # getting the final result
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        logger.debug("Processed final data")
        return train_processed_data,test_processed_data
    except Exception as e:
        print("Some error occured.")
        print(e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'processed')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.debug('file formed to the desired path : ',data_path)
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise

def main():
    train_data, test_data = get_data(train_data_path='./data/raw/train.csv',
                                    test_data_path='./data/raw/test.csv')
    train_data = normalize_text(train_data)
    test_data = normalize_text(test_data)
    train_processed_data, test_processed_data = get_final(train_data=train_data,test_data=test_data)
    save_data(train_processed_data,test_processed_data,data_path='data')

if __name__ == '__main__':
    main()