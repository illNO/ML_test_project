from pickle import load
import pandas as pd


def preprocess(data):
    scaler = load(open('scaler.pkl', 'rb'))
    X_test = scaler.transform(data)
    return X_test


def test(X_test):
    gbm = load(open('model/model.pkl', 'rb'))
    y_pred_gbm = gbm.predict(X_test)
    pd.DataFrame(y_pred_gbm).to_csv("data/results.csv", index=None)
    

if __name__ == "__main__":
    test_df = pd.read_csv('data/hidden_test.csv')

    X_test = preprocess(test_df)
    test(X_test)
