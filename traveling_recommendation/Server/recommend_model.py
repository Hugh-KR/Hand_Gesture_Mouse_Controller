import numpy as np
import tensorflow as tf


def recommend_place(input_data):

    model_path = 'model_final.h5'
    model = tf.keras.models.load_model(model_path)
    input_data = tf.convert_to_tensor(np.array(input_data).reshape(-1,22))
    pred = model.predict(input_data)[0][0]
    return pred


