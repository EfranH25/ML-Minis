import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
from sklearn import ensemble

def run(fold):
    df = pd.read_csv('F:\Work\Project Portfolio\PROJECT - ML Minis\Cat in the Dat\input\cat_train_fold.csv')

    features = [
        f for f in df.columns if f not in ('kfold', 'id', 'target')
    ]

    for col in features:
        df[col] = df[col].astype(str).fillna('NONE')

    for col in features:
        # initialize and label encode col
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])

        df[col] = lbl.transform(df[col])

    # get training and valid data w/ respective folds
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    # get training and validation data
    x_train = df_train[features].to_numpy()
    x_valid = df_valid[features].to_numpy()

    # initialize random forest model and fit model
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    model.fit(x_train, df_train['target'].to_numpy())

    # make prediction w/ validation and get auc score
    valid_pred = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid['target'], valid_pred)
    print(auc)

if __name__ == '__main__':
    for fold in range(5):
        run(fold)