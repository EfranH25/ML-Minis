import os
import argparse

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing

import config_file
import model_dispatcher

def run(fold, model):
    '''
    :param fold: select which fold to hold out as test set for cross validation
    :return: accuracy of selected model
    '''

    # read training data csv
    df = pd.read_csv(config_file.TRAINING_FILE)

    # selects the training set from df i.e. rows that don't equal selected fold
    df_train = df[df['kfold'] != fold].reset_index(drop=True)

    # selects test set from df i.e. rows that do equal selected fold
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    # sets x input of train set by dropping target label column and kfold column
    # sets y input of train set by only using label column
    # sets x and y to numpy arrays
    x_train = df_train.drop(['label', 'kfold'], axis=1).to_numpy()
    y_train = df_train['label'].to_numpy()

    # does same thing above but to the test set
    x_valid = df_valid.drop(['label', 'kfold'], axis=1).to_numpy()
    y_valid = df_valid['label'].to_numpy()

    # initialize model by fetching from model dispatcher
    clf = model_dispatcher.models[model]

    # fit model to training data
    clf = clf.fit(x_train, y_train)

    # create and score predictions
    preds = clf.predict(x_valid)

    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f'Fold={fold}, Accuracy={accuracy}, Model={model}' )

    # save model
    joblib.dump(clf,
                os.path.join(config_file.MODEL_OUTPUT, f'{model}_model_{fold}.bin')
                )

if __name__ == '__main__':
    print('START')
    # initialize and create argumentparser class of argparse to take in commands
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fold',
        type=int
    )
    parser.add_argument(
        '--model',
        type=str
    )

    args = parser.parse_args()

    # pass the inputs from argparser to run from command line
    run(fold=args.fold, model=args.model)

    print('press enter to complete')
    complete = input()