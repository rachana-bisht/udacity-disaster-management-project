# Disaster Response Pipeline Project README FILE

# Installation
Beyond the Anaconda distribution of Python, the following packages need to be installed for nltk:
* punkt
* wordnet
* stopwords

# Project Motivation
In this project, I appled data engineering, natural language processing, and machine learning skills to analyze message data from ‘FIGURE EIGHT’ that people sent during disasters. The purpose of the project is to build a model for an API that classifies disaster messages. Using the web app an emergency worker can input a new message and get classification results in several categories so to have an idea what kind of help is needed: "water", "shelter", "food", etc.


# File Descriptions
There are three main foleders:
1. data
    * disaster_categories.csv: dataset including all the categories
    * disaster_messages.csv: dataset including all the messages
    * process_data.py: ETL pipeline scripts to read, clean, and save data into a database
    * DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
    * train_classifier.py: machine learning pipeline scripts to train and export a classifier
    * classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer
3. app
    * run.py: Flask file to run the web application
    * templates contains html file for the web applicatin


# Licensing, Authors, Acknowledgements
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.


