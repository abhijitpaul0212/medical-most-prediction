import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


#Functions
def prepare_data(df):
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
    df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
    df.drop('region', inplace=True, axis=1)
    return df

def predict_cost(data, model):
    if type(data) == dict:
        df = pd.DataFrame(data)
    else:
        df = data
    prepared_df = prepare_data(df)
    y_pred=model.predict(prepared_df)
    return y_pred