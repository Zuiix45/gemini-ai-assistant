import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def createModel(modelIndex: int = 0, temparature: float = 0.9, top_p: float = 1, top_k: int = 1, max_output_tokens: int = 2048):
    genai.configure(api_key=GOOGLE_API_KEY)
    
    generation_config = genai.GenerationConfig(
        temperature=temparature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens
    )

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    model_name = getSupportedModels()[modelIndex].name.split("/")[1]
    
    print(f"Using model: {model_name}")

    return genai.GenerativeModel(model_name=model_name,
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def start_chat(model: genai.GenerativeModel):
    return model.start_chat(history=[])

def getSupportedModels():
    result = []
    
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            result.append(m)
            
    return result

async def sendMessage(chat: genai.ChatSession, message: str):
    return chat.send_message(message)
