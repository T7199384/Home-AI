import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

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
print(adl_numpy)

#converting lists to numpy arrays
sequences = np.array(sequences)

print(sequences)

activity_labels = np.array(activity_labels)


#splitting data into train, validate, and test
total_samples = len(sequences)
train_samples = int(0.8 * total_samples)
val_samples = int(0.1 * total_samples)

X_train = sequences[:train_samples]
y_train = activity_labels[:train_samples]

X_val = sequences[train_samples:train_samples + val_samples]
y_val = activity_labels[train_samples:train_samples + val_samples]

X_test = sequences[train_samples + val_samples:]
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
