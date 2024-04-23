import requests
import reflex as rx
import google.generativeai as genai
import os
import numpy as np
import cv2
import json

# Define the API key directly in the script for Google's API
API_KEY = "AIzaSyDGY_bjuq_LwQOfbo1rhFuvCLstU9Fv51E"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

# Define your Hugging Face API token
HUGGING_FACE_API_TOKEN = "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"

class QA(rx.Base):
    """A question and answer pair."""
    question: str
    answer: str

DEFAULT_CHATS = {
    "Intros": [],
}

class State(rx.State):
    """The app state."""
    img: list[str] = []
    chats: dict[str, list[QA]] = DEFAULT_CHATS
    current_chat: str = "Intros"
    processing: bool = False
    new_chat_name: str = ""

        
    async def process_question(self, form_data: dict[str, str]):
        question = form_data["question"]
        if question == "":
            return

        async for value in self.generate_response(question):
            yield value

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Process uploaded files and send them to the Hugging Face API."""
        for file in files:
            uploaded_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename

            with outfile.open("wb") as file_object:
                file_object.write(uploaded_data)

            self.img.append(str(outfile))

            # Prepare and send the image to the Hugging Face API
            api_response = await self.classify_brain_tumor(str(outfile))
            diagnosis = await self.generate_diagnosis(api_response)


    async def classify_brain_tumor(self, image_file_path_str):
        """Encode an image and send it to the brain tumor classification API."""
        # Read the image in binary format
        image_data = cv2.imread(image_file_path_str, cv2.IMREAD_COLOR)
        _, buffer = cv2.imencode('.jpg', image_data)
        binary_data = buffer.tobytes()

        # Define the API URL and headers
        BRAIN_TUMOR_API_URL = "https://api-inference.huggingface.co/models/Devarshi/Brain_Tumor_Classification"
        headers = {
            "Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"
        }

        # Send the POST request with the binary data
        response = requests.post(BRAIN_TUMOR_API_URL, headers=headers, data=binary_data)
        
        # Parse the response into a dictionary
        response_json = response.json()
        print(response_json)
        # Assume response_json is your list of dictionaries

        # Find the dictionary with the highest 'score'
        highest_scoring_dict = max(response_json, key=lambda x: x['score'])

        # Extract the label of the highest scoring dictionary
        highest_scoring_label = highest_scoring_dict['label']

        prediction = response_json
        return prediction

    def add_response_to_chat(self, question, answer):
        """Add a new QA pair to the current chat with the response from the API."""
        qa = QA(question=question, answer=answer)
        self.chats[self.current_chat].append(qa)
        self.chats = self.chats  # Trigger an update

    def create_chat(self):
        """Create a new chat."""
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []
        self.new_chat_name = ""  # Reset the new chat name

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat."""
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles."""
        return list(self.chats.keys())
    
    async def generate_response(self, question: str):
        """Process the question using Google's generative AI API."""
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)
        self.processing = True
        yield

        full_prompt = ("You are MIRAI, a Medical AI assistant whose sole focus is on brain tumors, its symptoms, " +
                       "and courses of action that can help cure this disease. You cannot answer questions that " +
                       "are not related to brain tumors, its symptoms, or its cures." + question)
    
        # Instantiate the model
        
        # Generate the content
        response = model.generate_content(full_prompt)
    
        
        # Retrieve the text from the response
        answer_text = response.text if response.text else ""
        self.chats[self.current_chat][-1].answer += answer_text
        self.chats = self.chats
        yield

        self.processing = False

    async def generate_diagnosis(self, prediction: str):
        """Process the question using Google's generative AI API."""
        qa = QA(question="Uploaded MRI", answer="")
        self.chats[self.current_chat].append(qa)
        self.processing = True

        full_prompt = ("Your role is Brain Tumor Disease Expert. Now I will provide you with the patient diagnosis prediction. Please format it professionally, as you are a Brain Tumor Disease Expert now! Let them know that mirAI is a finetuned AI model for classifying MRI Images of the brain, and that mirAI can sometimes be wrong!" +
                    "In your diagnosis, do not include patient name. First check if the user query is related to Brain Tumor or not. If you are asked about the user's If you are asked  If it is not a" +
                    "Brain Tumor then simply explain that mirAI is an AI that is trained on MRI images and it is out of your scope to diagnose anything other than brain tumors and its symptoms." +
                    "The prediction is given in the json format, diagnose the patient with percent chances of the user having these diseases: " + json.dumps(prediction))

        # Instantiate the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate the content
        response = model.generate_content(full_prompt)  # make sure to await the response

        # Retrieve the text from the response
        answer_text = response.text if response.text else ""
        
        # Here you can remove the "Diagnosis: no_tumor" part from the answer_text if it is included
        # You can use str.replace() or any other string manipulation method
        # For example, if 'Diagnosis: ' always appears at the start:
        diagnosis_label = f"{prediction}"
        if answer_text.startswith(diagnosis_label):
            answer_text = answer_text[len(diagnosis_label):].strip()

        # Append the clean answer text to the chat
        self.chats[self.current_chat][-1].answer += answer_text
        self.chats = self.chats  # Trigger an update if necessary

        self.processing = False
        return answer_text  # Return the final answer text


