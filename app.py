from flask import Flask, jsonify, request
import pickle
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tensorflow import keras


def get_scaler(scalerPath):
    return pickle.load(open(scalerPath, 'rb'))

def transformData(data_frame):
    globalEnergyT = data_frame['global energy'].transform([np.cbrt])
    relativeHumidityT = data_frame['rel. hum.'].transform([np.square])

    data_frame['global energy'] = globalEnergyT
    data_frame['rel. hum.'] = relativeHumidityT

    return data_frame


def predictMissingData(data_frame):
    time_df = data_frame['time']
    data_frame.drop(['time'], axis=1, inplace=True)
    imp_mean = IterativeImputer()
    imp_mean.fit(data_frame)
    data_frame.iloc[:, :] = imp_mean.transform(data_frame)
    data_frame['time'] = time_df
    return data_frame


def prepeareData(data_frame):
    data_frame = predictMissingData(data_frame)
    data_frame.set_index(['time'], inplace=True)
    data_frame.sort_values(by=['time'], ascending=True, inplace=True)
    return transformData(data_frame[['diffusive energy', 'global energy', 'rel. hum.', 'precipitation']])


def scaleData(data, scaler):
    return scaler.transform(data)

def reverseScaleData(data, scaler):
    return scaler.inverse_transform(data)

def getXShapes(df_scaled, window_size):
    return np.array([df_scaled[i: i + window_size] for i in range(len(df_scaled) - window_size)])

def getRightArraySize(df_scaled, hops, window_size):
    test_X = df_scaled[:hops + window_size]
    list_test_X = getXShapes(test_X, window_size)
    return list_test_X.reshape(len(list_test_X), window_size, 4)

app = Flask(__name__)

@app.route('/api/predict', methods = ['POST'])
def predictRNN():
    jsonData = request.json
    x_scaler = get_scaler('./binarydata/x_scaler.pk1')
    y_scaler = get_scaler('./binarydata/y_scaler.pk1')
    rnnModel = keras.models.load_model('./binarydata/myRNN_model.h5')
    data_frame = prepeareData(pd.json_normalize(jsonData))
    scaled_data_frame = scaleData(data_frame, x_scaler)
    reshaped_data = getRightArraySize(scaled_data_frame, 1028, 144)
    predictedResult = reverseScaleData(rnnModel.predict(reshaped_data), y_scaler)
    jsonResult = {
        'prediction': float(predictedResult[0][0])
    }
    return jsonify(jsonResult)