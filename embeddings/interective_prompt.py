import openai
from dotenv import load_dotenv


load_dotenv()

client = openai.OpenAI()

fact_sheet_computer = """
OVERVIEW
- A computer targeted at gamers

HARDWARE
- Ryzen 5 processor
- 16 GB RAM
- Liquid cooling
- 1 TB SSD

SOFTWARE
- Windows
- Antivirus software
- RGB controller

ACCESSORIES
- 4k Display
- RGB keyboard and mouse
"""

prompt = f"""
You are an Ad Writer. Your task is to write a catchy ad based on the specifications
provided in the technical factsheet . the information for technical specifications is delimited by triple backticks.

Format everything as HTML that can be used in a website. 
Place the description in a <div> element.

Technical specifications: ```{fact_sheet_computer}```
"""

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

response = get_completion(prompt)
print(response)