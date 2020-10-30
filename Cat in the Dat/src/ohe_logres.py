import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    # get training data and training features
    df = pd.read_csv('F:\Work\Project Portfolio\PROJECT - ML Minis\Cat in the Dat\input\cat_train_fold.csv')
    features = [
        f for f in df.columns if f not in ['kfold', 'target', 'id']
    ]

    for col in features:
        df[col] = df[col].astype(str).fillna('NONE')

    # get training and validation folds, drop kfold column
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    # initialize one hot encoded
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    full_data = pd.concat([df_train[features], df_valid[features]])
    ohe.fit(full_data[features])

    # transform training and validation data
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    # initialize model
    model = linear_model.LogisticRegression()

    # fit model
    model.fit(x_train, df_train['target'].to_numpy())

    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s (since predict_proba has multiple columns for each outcome)
    valid_preds = model.predict_proba(x_valid)[:, 1]

    print(model.predict_proba(x_valid)[:, 1])
    print(model.predict_proba(x_valid))

    # get roc and auc score
    auc = metrics.roc_auc_score(df_valid['target'].to_numpy(), valid_preds)

    # print auc
    print(auc)

if __name__ == '__main__':
    print('start')
    run(0)
