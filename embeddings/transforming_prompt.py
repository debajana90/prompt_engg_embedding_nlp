"""Transforming
Large language models can also be used in Transforming tasks, such as

Translation
Translating text from one language to one or many languages.
The translated text can be either formal or informal way of speaking.
Spelling and grammar correction
Ask the model to proofread a text which you're not sure is grammatically correct and output the corrected text
Format transformation
Providing HTML and getting the equivalent JSON format."""

import openai
from dotenv import load_dotenv


load_dotenv()

client = openai.OpenAI()

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

translate_text = """
The sun rises in the east.
"""

prompt = f"""
Your task is to translate the given english text to spanish.

Give the translation in the following format
Spanish: ...

Text: ```{translate_text}```
"""

response = get_completion(prompt)
print(response)