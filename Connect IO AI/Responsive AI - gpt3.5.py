from gtts import gTTS
import pygame
import os
from openai import OpenAI

openai_client= OpenAI(
    api_key = "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")

def get_feedback(prompt):
    # Generate response from GPT-3 based on the prompt   
    response = openai_client.completions.create(
        model="gpt-3.5-turbo",  # You can choose another engine or model here
        prompt=prompt,
        max_tokens=150,  # Adjust this as needed
        temperature=0.7,  # Adjust this for creativity vs accuracy balance
        n=1,
        stop=None,
        timeout=None,
    )
    
    # Extract and return the generated text
    feedback=response.choices[0].text.strip()
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