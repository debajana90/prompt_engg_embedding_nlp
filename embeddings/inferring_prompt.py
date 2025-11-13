"""Large language models can also be used for inferring tasks such as

Sentiment analysis
Emotion recognition
Information extraction
Topic inference
The use cases of Inferring can be

Extract different types of information from a piece of text such as
Identifying the sentiment of a product review
Extracting the item and brand from a review
Determining the topics covered in a news article.
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
Identify a list of emotions that the writer of the following review is expressing. Include no more than five items in the list. Format your answer as a list of lower-case words separated by commas.

Review text: '''{prod_review}'''
"""

response = get_completion(prompt)
print(response)