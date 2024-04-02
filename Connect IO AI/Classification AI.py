import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

#getting dataset
adl_df = pd.read_csv('adl_dataset.csv')

#sensor data and duration extraction
sensor_data = adl_df['sensor_data'].values
durations = adl_df['duration(s)'].values

#making fixed length sequences to iterate over the dataset, this will take 100 records from each of the activities as iternations
sequence_length = 100
step = 100

sequences = []
activity_labels = []

for i in range(0, len(sensor_data) - sequence_length, step):
    sequence = sensor_data[i:i + sequence_length]
    activity_label = adl_df['activity'].iloc[i + sequence_length]
    sequences.append(sequence)
    activity_labels.append(activity_label)


adl_numpy = adl_df.to_numpy
#print(adl_numpy)

#converting lists to numpy arrays
sequences = np.array(sequences)

print(sequences)

activity_labels = np.array(activity_labels)


#preparing to convert sequences to features for neural network training
features_list = []

for sequence_str in sequences:
    activations = []
    current_sensor = None
    current_state = None
    current_duration = None
    
    sequence_str = str(sequence_str)
    
    #splits the sequence into sensor name, sensor state and time duration
    for token in sequence_str.split('; '):
        if token.startswith("t:"):
            current_duration = int(token.split(":")[1])
            activations.append((current_sensor, current_state, current_duration))
        else:
            parts = token.split(":")
            current_sensor = parts[0]
            current_state = ":".join(parts[1:])

    #creating a set for the unique sensors of each sequence
    unique_sensors = set(sensor for sensor, _, _ in activations)
    num_sensors = len(unique_sensors)
    
    #mapping sensor names to columns
    sensor_index_map = {sensor: i for i, sensor in enumerate(unique_sensors)}
    
    #preparing the feature array
    max_length = len(activations)
    features = np.zeros((max_length, num_sensors, 2))  # Each row contains state (ON/OFF) and duration
    
    #making the feature array
    for i, (sensor, state, duration) in enumerate(activations):
        sensor_index = sensor_index_map[sensor]
        features[i, sensor_index, 0] = 1 if state == "ON" else 0  # Encode state as 1 for ON, 0 for OFF
        features[i, sensor_index, 1] = duration
    
    features_list.append(features)
    
###for i, sequence in enumerate(features_list):
###    print(f"Shape of sequence at position {i}:", len(sequence))    

#converting features to numpy
features_np_array = np.array(features_list)

#printing features array
for i, features in enumerate(features_np_array):
    print(f"Sequence {i+1} features:")
    print(features)
    print()



#splitting data into train, validate, and test
total_samples = len(sequences)
train_samples = int(0.8 * total_samples)
val_samples = int(0.1 * total_samples)

X_train = features_np_array[:train_samples]
y_train = activity_labels[:train_samples]

X_val = features_np_array[train_samples:train_samples + val_samples]
y_val = activity_labels[train_samples:train_samples + val_samples]

X_test = features_np_array[train_samples + val_samples:]
y_test = activity_labels[train_samples + val_samples:]

#The number of activities there are in the dataset
num_classes=16

#creating the neural network
model = models.Sequential([
    layers.Conv1D(32,3,activation='relu',input_shape=(sequence_length,1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

#compliling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#printing the sumarry of the model
model.summary()

#training the neural network model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

#evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)
