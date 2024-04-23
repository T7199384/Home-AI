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

#determine unique sensors
unique_sensors = set()
for sequence in sequences:
    for sensor_sequence in sequence:
        unique_sensors.update(sensor.split(':')[0] for sensor in sensor_sequence.split('; '))
num_unique_sensors = len(unique_sensors)


#defining the RNN
model = tf.keras.Sequential([
    layers.Embedding(input_dim=num_unique_sensors, output_dim=64),
    layers.LSTM(32, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])


#model compiling
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
'''''
#model training
model.fit(dataset, epochs=20)
'''''
for batch in dataset:
    input_data, labels = batch
    # Convert TensorFlow tensors to NumPy arrays for easier inspection
    input_data_np = input_data.numpy()
    labels_np = labels.numpy()
    
    # Print or inspect the NumPy arrays
    print("Input data shape:", input_data_np.shape)
    print("Labels shape:", labels_np.shape)
    # Print or inspect more details as needed
    print("Input data:", input_data_np)
    print("Labels:", labels_np)