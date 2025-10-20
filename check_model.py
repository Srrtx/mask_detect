import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('TM80keras_model.h5')
model.summary()