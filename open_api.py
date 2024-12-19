import base64

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI()  # api_key=OPENAI_API_KEY


def ask_gpt(messages, model):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=4096,
        top_p=1
    )
    return response.choices[0].message.content
