"""
Summarizing is the process to summarize text, which can be useful for quickly glancing over a large volume of text.
"""

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


prod_review = """
I recently purchased a 10-pack of Amazon Basics AA High-Performance Alkaline Batteries, and I'm very happy with them. They arrived quickly with free shipping for Prime members, and they've been working great in all of my devices. I've been using them in my remote controls, flashlights, and toys, and they've all been lasting a long time. I'm impressed with the quality of these batteries, and I would definitely recommend them to others. They're a great value for the price, and they're reliable and long-lasting. I'm glad I made the switch to Amazon Basics batteries.
"""

prompt = f"""
Your task is to generate a short summary of a product review from an ecommerce site. 

Summarize the review below, delimited by triple 
backticks, in exactly 10 words. 

Review: ```{prod_review}```
"""

response = get_completion(prompt)
print(response)