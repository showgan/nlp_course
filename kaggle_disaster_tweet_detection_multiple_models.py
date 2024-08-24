import numpy as np
import pandas as pd
import time
import pickle
import preprocess_kgptalkie as ps
import spacy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix,  precision_score, recall_score
from lazypredict.Supervised import LazyClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

"""
I used the following resources as references:
Competition page:
https://www.kaggle.com/competitions/nlp-getting-started/overview

Classical ML + deep learning code examples from the Udemy course.

Basic text clean-up code from the Udemy course (package name: preprocess_kgptalkie).

How to run in batch mode:
python kaggle_disaster_tweet_detection_multiple_models.py > & kaggle_disaster_tweet_detection_multiple_models.log

"""


###########################
# DATA PROCESSING METHODS #
###########################
def load_data(base_path: str) -> (pd.DataFrame, pd.DataFrame):
    print(f'-I- Loading training and test datasets from {base_path}/train.csv and'
          f' {base_path}/test.csv.')
    prev_time = time.time()
    df_train = pd.read_csv(f'{base_path}/train.csv')
    df_test = pd.read_csv(f'{base_path}/test.csv')
    df_train.head()
    print(f'-I- Datasets loading time: {time.time() - prev_time:.1f} seconds')
    return df_train, df_test


def clean_text(df_list: list, text_column: str):
    print(f'-I- Cleaning text in training and test datasets.')
    global spacy_nlp
    # prev_time = time.time()
    # df[text_column] = df[text_column].apply(lambda x: str(x).lower())
    # print(f'lower: {time.time() - prev_time:.1f} seconds')
    start_time = time.time()
    for df in df_list:
        prev_time = start_time
        df[text_column] = df[text_column].apply(lambda x: ps.cont_exp(x))
        print(f'-I- Time for word abbreviation expansions: {time.time() - prev_time:.1f} seconds')
        prev_time = time.time()
        df[text_column] = df[text_column].apply(lambda x: ps.remove_emails(x))
        print(f'-I- Time for removing email addresses: {time.time() - prev_time:.1f} seconds')
        prev_time = time.time()
        df[text_column] = df[text_column].apply(lambda x: ps.remove_html_tags(x))
        print(f'-I- Time for removing HTML tags: {time.time() - prev_time:.1f} seconds')
        prev_time = time.time()
        df[text_column] = df[text_column].apply(lambda x: ps.remove_urls(x))
        print(f'-I- Time for removing URLs: {time.time() - prev_time:.1f} seconds')
        prev_time = time.time()
        df[text_column] = df[text_column].apply(lambda x: ps.remove_special_chars(x))
        print(f'-I- Time for removing special characters: {time.time() - prev_time:.1f} seconds')
        prev_time = time.time()
        df[text_column] = df[text_column].apply(lambda x: ps.remove_accented_chars(x))
        print(f'-I- Time for converting UTF characters ("accented characters") to ASCII:'
              f' {time.time() - prev_time:.1f} seconds')
        prev_time = time.time()
        df[text_column] = df[text_column].apply(lambda x: ps.make_base(x, spacy_nlp=spacy_nlp))
        print(f'-I- Time for lemmatization: {time.time() - prev_time:.1f} seconds')
        # TODO disable spelling correction since it takes a long time
        #  (47 minutes for train + test datasets)
        print('-W- Skipping spelling correction to save time during development.')
        # prev_time = time.time()
        # df[text_column] = df[text_column].apply(lambda x: ps.spelling_correction(x).raw_sentences[0])
        # print(f'-I- Time for spelling corrections: {time.time() - prev_time:.1f} seconds')
    print(f'-I- Total time for cleaning text: {time.time() - start_time:.1f} seconds')


def convert_text_to_vector(text: str) -> list:
    global spacy_nlp
    doc = spacy_nlp(text)
    return doc.vector


def add_vectors(df_list: list, text_column: str, vector_column: str):
    print(f'-I- Adding vectors to training and test datasets.'
          f' Source text column: "{text_column}". Target vector column "{vector_column}".')
    start_time = time.time()
    for df in df_list:
        prev_time = start_time
        df[vector_column] = df[text_column].apply(lambda text: convert_text_to_vector(text))
        print(f'-I- Time for converting texts to vectors: {time.time() - prev_time:.1f} seconds')
    print(f'-I- Total time for converting texts to vectors: {time.time() - start_time:.1f} seconds')


def get_X(df_list: list, vector_column: str) -> list:
    print(f'-I- Getting X matrices of training and test datasets from'
          f' vectors in column: "{vector_column}".')
    X_list = []
    start_time = time.time()
    for df in df_list:
        prev_time = start_time
        X = df[vector_column].to_numpy()
        X = X.reshape(-1, 1)
        X = np.concatenate(np.concatenate(X, axis=0), axis=0).reshape(-1, 300)
        X_list.append(X)
        print(f'-I- Time: {time.time() - prev_time:.1f} seconds')
    print(f'-I- Total time for getting X matrices: {time.time() - start_time:.1f} seconds')
    return X_list


##############################
# PREPROCESSING AND CLEANING #
##############################
start_time_overall = time.time()
start_time_preprocessing = start_time_overall

base_path = '/home/haroon/Learning/Intel_AI_Certificate_2024/NLP/project'
# Load training and test datasets:
df_train, df_test = load_data(base_path)
# TODO For testing - use smaller dataset to save time
# print('-W- Reducing dataset size to save time during development.')
# df_train=df_train.head(500)
# df_test=df_test.head(150)

# Save original train and test datasets:
# df_train.to_csv(f'{base_path}/train_before_cleaning.csv', index=False)
# df_test.to_csv(f'{base_path}/test_before_cleaning.csv', index=False)

# Load the large English NLP model of Spacy:
prev_time = time.time()
spacy_nlp = spacy.load('en_core_web_lg')
print(f'-I- Time for loading the large English NLP model of Spacy:'
      f' {time.time() - prev_time:.1f} seconds')

# Clean texts:
clean_text(df_list=[df_train, df_test], text_column='text')
# Save clean train and test datasets:
df_train.to_csv(f'{base_path}/train_after_cleaning.csv', index=False)
df_test.to_csv(f'{base_path}/test_after_cleaning.csv', index=False)
print(f'-I- Clean train dataset sample:\n{df_train.head()}')

# Generate word vectors:
vector_column = 'vector'
add_vectors(df_list=[df_train, df_test], text_column='text', vector_column=vector_column)
print(f'-I- Preprocessed train dataset shape and sample:\n{df_train.shape}\n{df_train.head()}')

# Get the X matrices:
X_train, X_test = get_X(df_list=[df_train, df_test], vector_column=vector_column)
print(f'-I- Shape of the X_train matrix (before splitting to train and validation): {X_train.shape}')

# Get the y labels of the training set (ground truths):
y_train = df_train['target']

# Split the training set to train and validation. Make sure both splits have balanced labels:
prev_time = time.time()
X_train_split, X_validation_split, y_train_split, y_validation_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)
print(f'-I- Time for spliting training dataset to train and validation:'
      f' {time.time() - prev_time:.1f} seconds')
print(f'-I- Shapes of train and validation splits: {X_train_split.shape}, {X_validation_split.shape}')

print(f'-I- Total time for preprocessing and cleaning:'
      f' {time.time() - start_time_preprocessing:.1f} seconds')

##################################
# MODEL TRAINING AND INFERENCING #
##################################

####################
# TRAINING METHODS #
####################


def save_submission(df_test: pd.DataFrame, predictions: np.ndarray, file_name: str):
    print(f'-I- Saving test predictions to {file_name}')
    df = pd.DataFrame()
    df['id'] = df_test['id']
    df['target'] = pd.DataFrame(predictions)
    # df.columns = ['id', 'target']
    # df_test.to_csv(f'{base_path}/temp.csv', index=False)
    df.to_csv(file_name, index=False)


def run_logistic_regression(X_train_split, X_validation_split,
                            y_train_split, y_validation_split,
                            X_test):
    print('-I- Running Logistic Regression...')
    start_time = time.time()
    # Logistic Regression classification model from SciKit-Learn:
    classifier_lg = LogisticRegression(solver='liblinear')
    # Train:
    classifier_lg.fit(X_train_split, y_train_split)
    # Predict for the validation split:
    y_validation_split_pred = classifier_lg.predict(X_validation_split)
    # Print classification report for the validation split:
    print(classification_report(y_validation_split, y_validation_split_pred))
    # Save the trained model to a pickle file:
    pickle.dump(classifier_lg, open(f'{base_path}/logistic_regression_model.pkl', 'wb'))
    # Predict for the test set and save to file:
    y_test_pred = classifier_lg.predict(X_test)
    save_submission(
        df_test=df_test,
        predictions=y_test_pred,
        file_name=f'{base_path}/logistic_regression_test_predictions.csv')
    print(f'-I- Total time for training and predicting with Logistic Regression:'
          f' {time.time() - start_time:.1f} seconds')


def run_linear_svc(X_train_split, X_validation_split,
                   y_train_split, y_validation_split,
                   X_test):
    print('-I- Running Linear SVC...')
    start_time = time.time()
    # Linear Support Vector Classification model from SciKit-Learn:
    classifier_lsvc = LinearSVC(dual=True, max_iter=100000)
    # Train:
    classifier_lsvc.fit(X_train_split, y_train_split)
    # Predict for the validation split:
    y_validation_split_pred = classifier_lsvc.predict(X_validation_split)
    # Print classification report for the validation split:
    print(classification_report(y_validation_split, y_validation_split_pred))
    # Save the trained model to a pickle file:
    pickle.dump(classifier_lsvc, open(f'{base_path}/linear_svc_model.pkl', 'wb'))
    # Predict for the test set and save to file:
    y_test_pred = classifier_lsvc.predict(X_test)
    save_submission(
        df_test=df_test,
        predictions=y_test_pred,
        file_name=f'{base_path}/linear_svc_test_predictions.csv')
    print(f'-I- Total time for training and predicting with Linear SVC:'
          f' {time.time() - start_time:.1f} seconds')


def run_grid_search_cross_validation(
        X_train_split, X_validation_split,
        y_train_split, y_validation_split,
        X_test):
    print('-I- Running Grid Search Cross Validation with Logistic Regression...')
    start_time = time.time()
    # Use Grid Search Cross Validation with Logistic Regression model from Scikit-learn:
    logit = LogisticRegression(solver='liblinear')
    # Specify the desired hyperparameters (can be extended with other ranges and other parameters):
    hyperparameters = {
        'penalty': ['l1', 'l2'],
        'C': (1, 2, 3, 4)
    }
    # The Grid Search object:
    classifier_gs = GridSearchCV(estimator=logit, param_grid=hyperparameters, n_jobs=-1, cv=5)
    # Train with the different hyperparameters to find the best set of hyperparameters:
    classifier_gs.fit(X_train_split, y_train_split)
    # Print the best hyperparameters and the best score:
    print(f'-I- Best hyperparameters for Logistic Regression:\n{classifier_gs.best_params_}\n'
          f'-I- Best score: {classifier_gs.best_score_}')
    # Predict for the validation split using the best hyperparameters:
    y_validation_split_pred = classifier_gs.predict(X_validation_split)
    # Print classification report for the validation split:
    print(classification_report(y_validation_split, y_validation_split_pred))
    # Save the trained model to a pickle file:
    pickle.dump(classifier_gs, open(f'{base_path}/classifier_grid_search_cross_validation_model.pkl', 'wb'))
    # Predict for the test set and save to file:
    y_test_pred = classifier_gs.predict(X_test)
    save_submission(
        df_test=df_test,
        predictions=y_test_pred,
        file_name=f'{base_path}/classifier_grid_search_cross_validation_test_predictions.csv')
    print(f'-I- Total time for training and predicting with Grid Search Cross Validation:'
          f' {time.time() - start_time:.1f} seconds')


def run_lazzy_classifier(X_train_split, X_validation_split,
                         y_train_split, y_validation_split,
                         X_test):
    print(f'-I- Running Lazzy Classifier...')
    start_time = time.time()
    # Use Lazzy Classifier to train a multiple set of models and compare their scores:
    classifier_lazzy = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    # Run the various models:
    models, predictions = classifier_lazzy.fit(
        X_train_split, X_validation_split,  y_train_split, y_validation_split)
    # Print the results:
    print(f'-I- Score results from Lazzy Classifier:\n-I- Models:\n{models}')
    # Save the trained model to a pickle file:
    pickle.dump(classifier_lazzy, open(f'{base_path}/classifier_lazzy_model.pkl', 'wb'))
    # TODO Should I save models and predictions too?
    print(f'-I- Total time for training and predicting with Lazzy Classifier:'
          f' {time.time() - start_time:.1f} seconds')
    # TODO add prediciting and saving for the test set


def run_deep_neural_network_classification1(X_train_split, X_validation_split,
                                            y_train_split, y_validation_split,
                                            X_test):
    print('-I- Running Classical Deep Neural Network...')
    start_time = time.time()
    # Build neural net model with 2 hidden layers with 128 nodes each:
    nn_model = Sequential()
    nn_model.add(Input(shape=(X_train_split.shape[1],)))
    nn_model.add(Dense(128, activation='relu'))
    nn_model.add(Dense(128, activation='relu'))
    nn_model.add(Dense(1, activation='sigmoid'))
    print(nn_model.summary())
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    # Scale the data for standardization:
    scaler = MinMaxScaler()
    X_train_split_scaled = scaler.fit_transform(X_train_split)
    X_validation_split_scaled = scaler.transform(X_validation_split)
    # Train:
    epochs = 100
    nn_model.fit(X_train_split_scaled, y_train_split, batch_size=32, epochs=epochs)
    # Predict for the validation split:
    y_validation_split_pred = nn_model.predict(X_validation_split_scaled)
    # Finalize predictions from probability to class (0 or 1). Use a threshold of 0.6.
    # Note: The threshold might be not optimal - different values should be checked to find the best.
    y_validation_split_pred_final = y_validation_split_pred.squeeze()
    y_validation_split_pred_final = np.where(y_validation_split_pred_final >= 0.6, 1, 0)
    # Print confusion matrix for the validation split:
    print(confusion_matrix(y_validation_split, y_validation_split_pred_final))
    # Print classification report for the validation split:
    print(classification_report(y_validation_split, y_validation_split_pred_final))
    # Plot confusion matrix for the validation split:
    # TODO disabled plotting in batch mode
    # plot_confusion_matrix(confusion_matrix(y_validation_split, y_validation_split_pred_final))
    # plt.show()
    # Save the trained model to a pickle file:
    pickle.dump(nn_model, open(f'{base_path}/deep_neural_network_model.pkl', 'wb'))
    # Save the trained model to a keras formnat:
    nn_model.save(f'{base_path}/deep_neural_network_model.keras')
    # To load the trained model from a keras formnat:
    nn_model = load_model(f'{base_path}/deep_neural_network_model.keras')
    # Predict for the test set and save to file:
    X_test_scaled = scaler.fit_transform(X_test)
    y_test_pred = nn_model.predict(X_test_scaled)
    # Finalize predictions from probability to class (0 or 1). Use a threshold of 0.6:
    y_test_pred_final = y_test_pred.squeeze()
    y_test_pred_final = np.where(y_test_pred_final >= 0.6, 1, 0)
    save_submission(
        df_test=df_test,
        predictions=y_test_pred_final,
        file_name=f'{base_path}/deep_neural_network_test_predictions.csv')
    print(f'-I- Total time for training and predicting with Deep Neural Network:'
          f' {time.time() - start_time:.1f} seconds')


def plot_learning_curves(training_history, epochs):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, training_history.history['accuracy'])
    plt.plot(epoch_range, training_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    # Plot training & validation loss values
    plt.plot(epoch_range, training_history.history['loss'])
    plt.plot(epoch_range, training_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


# Run the different models:
start_time_training = time.time()
run_logistic_regression(X_train_split, X_validation_split, y_train_split, y_validation_split, X_test)
run_linear_svc(X_train_split, X_validation_split, y_train_split, y_validation_split, X_test)
run_grid_search_cross_validation(X_train_split, X_validation_split,
                                 y_train_split, y_validation_split, X_test)
run_lazzy_classifier(X_train_split, X_validation_split, y_train_split, y_validation_split, X_test)
run_deep_neural_network_classification1(X_train_split, X_validation_split,
                                        y_train_split, y_validation_split, X_test)
print(f'-I- Total time for all training and predicting:'
      f' {time.time() - start_time_training:.1f} seconds')

print(f'-I- Overall run time: {time.time() - start_time_overall:.1f} seconds')
