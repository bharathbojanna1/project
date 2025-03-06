import openai
from config import OPENAI_API_KEY

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

def ask_deepseek(query):
    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": query}]
    )
    return response["choices"][0]["message"]["content"]
