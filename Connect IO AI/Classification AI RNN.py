import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from kerastuner.tuners import RandomSearch
import importlib.util


adl_df = pd.read_csv('adl_dataset.csv')

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
    
ragged_encoded_sensor_data = tf.ragged.constant(encoded_sensor_data)
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

opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(dataset, epochs=20)



#parcial data on the laundry activity
partial_sensor_data = "sink_cabinet:ON; t:50; sink_cabinet:OFF; laundry_sensor:ON"

for i in range(5):
    encoded_partial_sequence = [hash(sensor) % 10000 for sensor in partial_sensor_data.split('; ')]

    ragged_partial_sequence = tf.ragged.constant([encoded_partial_sequence])
    dense_partial_sequence = ragged_partial_sequence.to_tensor(default_value=0)

    predicted_probabilities = model.predict(dense_partial_sequence)

    threshold = 0.5
    predicted_labels = (predicted_probabilities > threshold).astype(int)

    decoded_labels = label_encoder.inverse_transform(predicted_labels)

    print("Prediction", i+1, ":", decoded_labels)

#parcial data on the breakfast_cereal activity
partial_sensor_data = "cabinet_door:ON; t:21; cabinet_door:OFF; t:19; fridge_door:ON; t:77;"

for i in range(5):
    encoded_partial_sequence = [hash(sensor) % 10000 for sensor in partial_sensor_data.split('; ')]

    ragged_partial_sequence = tf.ragged.constant([encoded_partial_sequence])
    dense_partial_sequence = ragged_partial_sequence.to_tensor(default_value=0)

    predicted_probabilities = model.predict(dense_partial_sequence)

    threshold = 0.5
    predicted_labels = (predicted_probabilities > threshold).astype(int)

    decoded_labels = label_encoder.inverse_transform(predicted_labels)

    print("Prediction", i+1, ":", decoded_labels)
    
activity_messages = {
"cleaning_dishes": "Tips for cleaning dishes",
"laundry": "How can I improve washing my clothes in a washing machine?",
"cleaning": "tips on cleaning up the house faster",
"gardening": "tips on gardening",
"making_snack": "what is a good quick snack to eat?",
"reading": "Can you recommend any books?",
"working": "I'm working, how can I stay productive?",
"exercising": "Tips for exercising on a treadmill",
"watching_tv": "What should I watch?",
"relaxing": "I am relaxing, what is a good way to relax",
"sleeping": "I'm going to sleep, please do not disturb me",
"bathroom_break": "I am using the bathroom",
"personal_hygiene": "I am going to shower",
"dinner_pasta": "Tips for cooking pasta",
"lunch_sandwich": "Good sandwich ideas",
"breakfast_cereal": "how can I improve my breakfast cereal?",
}

decoded_labels_str = decoded_labels[0]
print(decoded_labels_str)

message = activity_messages[decoded_labels_str]
print(message)

#import responsive AI script
script_name = "Responsive AI.py"
module_name = "Responsive AI"
spec = importlib.util.spec_from_file_location(module_name, script_name)
script_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script_module)

script_module.run(message)