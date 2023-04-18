# Regression problem for unknown data

There are 3 main components:
1. `Exploration.ipynb` contains exploratory analysis of the dataset, basic training and metrics calculation.
2. `train.py` takes as input file name - dataset location and contains functions to preprocess data and train LightGBM model.
3. `test.py` takes raw data as input and contains functions to scale data using scaler from training dataset. Also it writes predictions to `data/results.csv`

---

To update the model run script `train.py` with new data and then make predictions running script `predict.py`
