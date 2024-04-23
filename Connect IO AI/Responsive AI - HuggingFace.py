from unittest.util import _MAX_LENGTH
from gtts import gTTS
import pygame
import os
from transformers import pipeline

text_generator = pipeline("text-generation", model="gpt2")

def get_feedback(prompt):
    # Generate response from GPT-3 based on the prompt   
    feedback = text_generator(prompt, max_length=1000, num_return_sequences=1, truncation=True, pad_token_id=text_generator.tokenizer.eos_token_id)[0]['generated_text']
    
    # Extract and return the generated text
    print(feedback)
    text_to_speech(feedback)

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
    prompt = input("Enter text: ")
    prompt_strip=prompt.strip()
    if prompt_strip and  any(char.isalnum() for char in prompt_strip):
        get_feedback(prompt)