import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import transformers


previously_trained_model = tf.keras.models.load_model('trained_model')

previously_trained_model.summary()
