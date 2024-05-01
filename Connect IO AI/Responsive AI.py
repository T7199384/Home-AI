from unittest.util import _MAX_LENGTH
from gtts import gTTS
import pygame
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = TFGPT2LMHeadModel.from_pretrained("gpt2", from_pt=True)
text_generator = pipeline("text-generation", model="gpt2")

def feedback_training(feedbacks):

    # Prepare the Data
    # Assuming feedbacks is a list of strings 

    # Tokenize the feedbacks and convert them into numerical representations
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(feedbacks)
    sequences = tokenizer.texts_to_sequences(feedbacks)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences)

    # Define the Model
    kmeans = KMeans(n_clusters=1)  # Adjust the number of clusters as needed

    # Fit the Model
    kmeans.fit(padded_sequences)

    # Get the Centroids (Representative Feedbacks)
    representative_feedbacks = kmeans.cluster_centers_
    
    # Compute the pairwise cosine similarity between all representative feedbacks
    similarities_matrix = cosine_similarity(padded_sequences)

    # Compute the average similarity of each feedback with all other feedbacks
    average_similarities = np.mean(similarities_matrix, axis=1)

    # Find the index of the feedback with the highest average similarity
    best_feedback_index = np.argmax(average_similarities)
    best_feedback = feedbacks[best_feedback_index]

    return best_feedback

def get_feedback(prompt):
    # Generate response from GPT-3 based on the prompt   
    feedbacks = []
    for _ in range(10):
        # Use the generate method from the GPT-2 model to generate text
        feedback = text_generator(prompt, max_length=1000, num_return_sequences=1, truncation=True, pad_token_id=text_generator.tokenizer.eos_token_id)[0]['generated_text']
        feedbacks.append(feedback)
        
    trained_feedback = feedback_training(feedbacks)
    
    # Extract and return the generated text
    print(trained_feedback)
    text_to_speech(trained_feedback)

def text_to_speech(text):
    # Create a gTTS object
    tts = gTTS(text=text, lang='en-uk')

    # Save the speech as an MP3 file
    tts.save("output.mp3")

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the MP3 file
    pygame.mixer.music.load("output.mp3")

    # Play the MP3 file
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Adjust the playback speed

    # Stop the music playback
    pygame.mixer.music.stop()

    # Close the pygame mixer
    pygame.mixer.quit()

    # Remove the temporary MP3 file
    os.remove("output.mp3")

while True:
    """
    prompt = input("Enter text: ")
    prompt_strip=prompt.strip()
    if prompt_strip and  any(char.isalnum() for char in prompt_strip):
        get_feedback(prompt)
    """
    print("Enter text: Cooking steak tips")
    get_feedback("cooking steak tips")