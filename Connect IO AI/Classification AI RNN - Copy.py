import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from kerastuner.tuners import RandomSearch

#getting dataset
adl_df = pd.read_csv('adl_dataset.csv')

#sensor data and duration extraction
sensor_data = adl_df['sensor_data'].values
durations = adl_df['duration(s)'].values

#encoding labels
label_encoder = LabelEncoder()
activity_labels = label_encoder.fit_transform(adl_df['activity'])

#creating hyperparameter search space
hyperparameters = {
    'learning_rate': [0.001, 0.01, 0.1],
    'num_units': [16, 32, 64],
    'batch_size': [16, 32, 64]
}


#making fixed length sequences to iterate over the dataset, this will take 100 records from each of the activities as iternations
sequence_length = 100
step = 100
sequences = []
activity_labels_split = []

for i in range(0, len(sensor_data) - sequence_length, step):
    sequence = sensor_data[i:i + sequence_length]
    sequences.append(sequence)
    activity_label = activity_labels[i + sequence_length]
    activity_labels_split.append(activity_label)

sequences = np.array(sequences)
activity_labels_split = np.array(activity_labels_split)

#preparing RNN for input

#encoding sensor data to integers
encoded_sensor_data = []
for sequence in sequences:
    encoded_sequence = [hash(sensor) % 10000 for sensor in str(sequence).split('; ')]
    encoded_sensor_data.append(encoded_sequence)
    
#creating raggedtensor to pad it as a densetensor
ragged_encoded_sensor_data = tf.ragged.constant(encoded_sensor_data)

#ragged to dense to make a rectangular tensor
dense_encoded_sensor_data = ragged_encoded_sensor_data.to_tensor(default_value=0)

#converting to tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices((dense_encoded_sensor_data, activity_labels_split))

#shuffling and batching the data
batch_size = 32
dataset = dataset.shuffle(buffer_size=len(encoded_sensor_data)).batch(batch_size)

#defining the RNN
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=32, mask_zero=True),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


#model compiling
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#model training
model.fit(dataset, epochs=10)
