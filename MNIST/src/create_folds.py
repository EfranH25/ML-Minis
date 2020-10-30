import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection

def kfold_val(df, k = 5):
    '''
    :param df: pandas data frame
    :param k: k value for number of folds
    :return: randomizes data and creates a kfold column --> used for kfold cross validation
    '''

    # creates new column for kfold
    df['kfold'] = -1
    # ranomizes rows of data
    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=k)
    for fold, (trn_,val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    df.to_csv('mnist_train_folds.csv')

def strat_kfold_val(df, k):
    '''
    :param df: pandas data frame
    :param k: k value for number of folds
    :return: randomizes data and creates a kfold column --> used for stratified kfold cross validation
    '''

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # randmoize rows of data
    df = df.sample(frac=1).reset_index(drop=True)

    # set target y value
    y = df['quality']

    # initialize stratified 5 fold
    kf = model_selection.StratifiedKFold(n_splits=k)

    # fill kfold columns with values
    # this kf needs X and y vals (y val to maintain class ratio in folds)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # save to new csv
    df.to_csv('strat_train_folds.csv', index=False)

if __name__ == '__main__':
    df = pd.read_csv('F:\Work\Project Portfolio\PROJECT - ML Minis\MNIST\input\mnist_train.csv')

    # from exploring data, we know mnist target value, label, is not skewed so we will use kfold
    # cross validation to split training data --> create 5 folds
    kfold_val(df, 5)