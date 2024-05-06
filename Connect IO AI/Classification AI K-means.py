import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from keras.preprocessing.sequence import pad_sequences
import importlib.util

#getting dataset
adl_df = pd.read_csv('adl_dataset.csv')

#sensor data and duration extraction
sensor_data = adl_df['sensor_data'].values

#data preparing for k-means
encoded_sensor_data = []
max_length = 0
for sequence in sensor_data:
    encoded_sequence = [hash(sensor) % 10000 for sensor in str(sequence).split('; ')]
    encoded_sensor_data.append(encoded_sequence)
    max_length = max(max_length, len(encoded_sequence))

encoded_sensor_data_padded = pad_sequences(encoded_sensor_data, dtype='float32', padding='post')

#preparing k-means for input

#padding sequences
encoded_sensor_data_padded = pad_sequences(encoded_sensor_data, dtype='float32', padding='post')

#fitting to k-means
num_activities = len(adl_df['activity'].unique())
kmeans = KMeans(n_clusters=num_activities, random_state=42)
kmeans.fit(encoded_sensor_data_padded)
    
#defining the cluster for activity mapping
cluster_labels = kmeans.labels_
activity_labels = adl_df['activity'].values
cluster_to_activity_mapping = {cluster_label: activity_label for cluster_label, activity_label in zip(cluster_labels, activity_labels)}



#parcial data on the laundry activity
partial_sensor_data = "sink_cabinet:ON; t:50; sink_cabinet:OFF; laundry_sensor:ON"

for i in range(5):
# Encode partial sensor data
    encoded_partial_sequence = [hash(sensor) % 10000 for sensor in partial_sensor_data.split('; ')]

    # Pad the partial sequence to match the maximum sequence length
    max_sequence_length = max([len(sequence) for sequence in encoded_sensor_data])
    padded_encoded_partial_sequence = np.pad(encoded_partial_sequence, (0, max_sequence_length - len(encoded_partial_sequence)), mode='constant')

   ## Fitting KMeans with explicit data type
    kmeans = KMeans(n_clusters=num_activities, random_state=42)
    kmeans.fit(encoded_sensor_data_padded.astype(np.float64))

    # Convert the input data to double precision floating-point numbers
    padded_encoded_partial_sequence = padded_encoded_partial_sequence.astype(np.float64)
    
    # Convert to contiguous array
    reshaped_sequence = np.ascontiguousarray(padded_encoded_partial_sequence)

    # Predict cluster label for reshaped partial sequence
    predicted_cluster_label = kmeans.predict([reshaped_sequence])[0]

    # Map cluster label to activity label
    predicted_activity_label = cluster_to_activity_mapping[predicted_cluster_label]

    print("Prediction", i+1, ":", predicted_activity_label)

#parcial data on the breakfast_cereal activity
partial_sensor_data = "cabinet_door:ON; t:21; cabinet_door:OFF; t:19; fridge_door:ON; t:77;"

for i in range(5):
    # Encode partial sensor data
    encoded_partial_sequence = [hash(sensor) % 10000 for sensor in partial_sensor_data.split('; ')]

    # Pad the partial sequence to match the maximum sequence length
    max_sequence_length = max([len(sequence) for sequence in encoded_sensor_data])
    padded_encoded_partial_sequence = np.pad(encoded_partial_sequence, (0, max_sequence_length - len(encoded_partial_sequence)), mode='constant')

   ## Fitting KMeans with explicit data type
    kmeans = KMeans(n_clusters=num_activities, random_state=42)
    kmeans.fit(encoded_sensor_data_padded.astype(np.float64))

    # Convert the input data to double precision floating-point numbers
    padded_encoded_partial_sequence = padded_encoded_partial_sequence.astype(np.float64)
    
    # Convert to contiguous array
    reshaped_sequence = np.ascontiguousarray(padded_encoded_partial_sequence)

    # Predict cluster label for reshaped partial sequence
    predicted_cluster_label = kmeans.predict([reshaped_sequence])[0]

    # Map cluster label to activity label
    predicted_activity_label = cluster_to_activity_mapping[predicted_cluster_label]

    print("Prediction", i+1, ":", predicted_activity_label)
    
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

# Convert decoded_labels to a string
decoded_labels_str = predicted_activity_label
print(decoded_labels_str)

# Accessing message for a specific activity label
message = activity_messages[decoded_labels_str]
print(message)

#import responsive AI script
script_name = "Responsive AI.py"
module_name = "Responsive AI"
spec = importlib.util.spec_from_file_location(module_name, script_name)
script_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script_module)

script_module.run(message)