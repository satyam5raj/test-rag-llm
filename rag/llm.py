# import openai
# import os

# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def ask_llm(prompt: str):
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2
#     )
#     return response.choices[0].message.content



# Updated llm.py for Google Gemini
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def ask_llm(prompt: str):
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,  # More creative responses
                max_output_tokens=500,  # Longer responses
                top_p=0.9
            )
        )
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"
