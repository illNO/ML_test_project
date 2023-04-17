import lightgbm as lgb

from datetime import datetime

from sklearn.metrics import r2_score, mean_squared_error, max_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pickle import dump, load
import pandas as pd


def preprocess_train(data):
    y = data['target']
    X = data.drop(columns='target')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=46)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    dump(scaler, open('scaler.pkl', 'wb'))
    # scaler = load(open('scaler.pkl', 'rb'))
    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):
    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1', 'l2'],
        #     'learning_rate': 0.0001,
        #     'feature_fraction': 0.9,
        #     'bagging_fraction': 0.7,
        #     'bagging_freq': 10,
        #     'verbose': 0,
        "max_depth": 16,
        "num_leaves": 128,
        "max_bin": 512,
        "num_iterations": 500
    }

    gbm = lgb.LGBMRegressor(**hyper_params)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            early_stopping_rounds=1000
            )

    dump(gbm, open(f'model/model.pkl', 'wb'))

    return gbm


if __name__ == "__main__":
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/hidden_test.csv')

    X_train, X_test, y_train, y_test = preprocess_train(train_df)
    model = train(X_train, X_test, y_train, y_test)
