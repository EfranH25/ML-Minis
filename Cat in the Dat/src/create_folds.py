import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    # get training data
    df = pd.read_csv('F:\Work\Project Portfolio\PROJECT - ML Minis\Cat in the Dat\input\cat_train.csv')

    # set kfold column to 01
    df['kfold'] = -1

    # randomize training data
    df = df.sample(frac=1).reset_index(drop=True)

    # get target value for strat kfold for equal distribution of targets in fold
    y = df['target'].to_numpy()

    # initialize kfold w/ 5 splits
    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    df.to_csv('cat_train_fold.csv', index=False)