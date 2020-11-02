import sys
# import libraries
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
  
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle



def load_data(database_filepath):
    db_name = 'sqlite:///'+ database_filepath
    engine = create_engine(db_name)
    df = pd.read_sql_table('disaster_response_new', con = engine)
    #Remove child_alone column, as it has all zeros only.
    df = df.drop(['child_alone'],axis=1)
    # As is shown aboe, label 2 in the 'related' column are neglible so it could be error. Replacing     2 with 1.
    # It will also make the 'related column as a binary label, 0 and 1.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    # Defining X (feature) and y (target variables)
    X = df['message']
   
    y = df.drop(['id','message','original','genre'], axis=1)
    
    category_name = y.columns.values
    y = y.astype(int)
    #category_name = y.columns.values
    print (category_name)
    
    return X, y, category_name
  
    
def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
 
    # take out all punctuation while tokenizing
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # lemmatize as shown in the lesson
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    
def build_model():
    
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mclf', MultiOutputClassifier(RandomForestClassifier()))
    ])   
# parameters set to this due to reduce the size of pkl file, which were too large (600MB) for uploading to github with my previous parameters.
    parameters = {
        'mclf__estimator__max_depth': [10, 50, None],
        'mclf__estimator__min_samples_leaf':[2, 5, 10],
    }
    model = pipeline
    #model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    #y_pred = pd.DataFrame(model.predict(X_test),
    #                 index = y_test.index,
    #                 columns = y_test.columns)
    
    y_pred = model.predict(X_test.values)
    class_report = classification_report(y_test.values, y_pred, target_names=category_names)
    return class_report
   


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
#        X, y, category_names = load_data(database_filepath)
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train.values, y_train.values)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()