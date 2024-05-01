from unittest.util import _MAX_LENGTH
from gtts import gTTS
import pygame
import os
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, DistributedType


class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.initial_length = len(texts)  # Store the initial length of the dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        print("Index:", idx)
        print("Length of texts:", len(self.texts))
        
        text = self.texts[idx]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = '[PAD]'
model = GPT2LMHeadModel.from_pretrained("gpt2")

sentence_bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

text_generator = pipeline("text-generation", model="gpt2")

# Move model to device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def feedback_training(feedbacks, model):
    # Fine-tune the model
    train_dataset = CustomDataset(feedbacks, tokenizer, 128)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        output_dir="./output",
    )
    
    # Initialize Accelerator
    accelerator = Accelerator()

    # Move model to accelerator device
    model, train_dataset, data_collator = accelerator.prepare(model, train_dataset, data_collator)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Debugging
    print("Length of training dataset:", len(train_dataset))
    print("Data collator:", data_collator)
    print("Training arguments:", training_args)
    print("Model:", model)
    print("Optimizer:", trainer.optimizer)

    try:
        print("Training starts...")
        trainer.train()
        print("Training completed.")
    except IndexError as e:
        print("Index error occurred during training loop:", e)
        print("Length of training dataset:", len(train_dataset))
    ###    print("Texts in dataset:", train_dataset.texts)

    # Get the last layer of the fine-tuned model for embeddings
    last_layer_model = model.transformer.h[-1]

    # Sample input sequence
    sample_input = "This is a test input sequence."

    # Tokenize the input
    tokenized_input = tokenizer(sample_input, return_tensors="pt")

    # Perform model inference
    with torch.no_grad():
        input_ids = tokenized_input.input_ids.to(model.device)
        output = last_layer_model(input_ids)

    # Inspect output tensor
    print("Output tensor shape:", output.shape)
    print("Output tensor data type:", output.dtype)
    print("Sample output values:", output[0, :10])  # Print the first 10 values as a sample


    # Get embeddings for all feedbacks
    feedback_embeddings = []
    for feedback in feedbacks:
        tokenized_input = tokenizer(feedback, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            
            input_ids = tokenized_input.input_ids.to(torch.int64)  # Convert to torch.int64
            input_ids = input_ids.to(model.device)  # Move input_ids to the device of the model
            print("Input IDs data type:", input_ids.dtype)
            print("Input IDs shape:", input_ids.shape)

            output = last_layer_model(input_ids)
            print("Output data type:", output.dtype)
            print("Output shape:", output.shape)

            print("Error occurred during model inference:", e)

            input_ids = tokenized_input.input_ids.to(torch.int64)  # Convert to torch.int64
            input_ids = input_ids.to(model.device)  # Move input_ids to the device of the model
            output = last_layer_model(input_ids)[0][:, 0, :].numpy()  # Ensure output tensor has consistent data type ###ERROR: mixed dtype (CPU): all inputs must share same datatype.
            feedback_embeddings.append(output)

    # Calculate similarity scores for each feedback against all other feedbacks
    similarity_scores = np.zeros((len(feedbacks), len(feedbacks)))
    for i in range(len(feedbacks)):
        for j in range(len(feedbacks)):
            similarity_scores[i, j] = np.dot(feedback_embeddings[i], feedback_embeddings[j].T) / (
                        np.linalg.norm(feedback_embeddings[i]) * np.linalg.norm(feedback_embeddings[j]))

    # Find the index of the feedback with the highest average similarity score
    avg_similarity_scores = np.mean(similarity_scores, axis=1)
    best_feedback_index = np.argmax(avg_similarity_scores)
    final_feedback = feedbacks[best_feedback_index]

    return final_feedback
def get_feedback(prompt, model):
    # Generate response from GPT-3 based on the prompt   
    feedbacks = []
    for _ in range(10):
        feedback = text_generator(prompt, max_length=1000, num_return_sequences=1, truncation=True, pad_token_id=text_generator.tokenizer.eos_token_id)[0]['generated_text']
        feedbacks.append(feedback)
        
    trained_feedback = feedback_training(feedbacks, model)
    
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
    get_feedback("cooking steak tips",model)