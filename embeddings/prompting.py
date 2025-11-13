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

text_to_get_topics_1 = f"""The computer consists of a keyboard, mouse and a display. The rest of the accessories should be purchased separately."""
text_to_get_topics_2 = f"""The movie is a well made piece of art with deep meaning on the subject of mental health."""
prompt = f"""
Given a text delimited by triple quotes. check If it contains texts based on computer, output the topics in the following format:

Topics - ...

If the text does not contain toics based on computer, then simply write \"No Topics on Computer.\"

\"\"\"{text_to_get_topics_2}\"\"\"
"""

response = get_completion(prompt)
print(response)