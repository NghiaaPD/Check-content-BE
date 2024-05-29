from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv
import google.generativeai as genai
import os
import torch
from IPython.display import Markdown
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()

load_dotenv()

model_phi_id = 'microsoft/phi-2'
model_phi = AutoModelForCausalLM.from_pretrained(model_phi_id)
tokenizer_phi = AutoTokenizer.from_pretrained(model_phi_id)

model_tinyllama_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model_tiny_llama = AutoModelForCausalLM.from_pretrained(model_tinyllama_id)
tokenizer_tiny_llama = AutoTokenizer.from_pretrained(model_tinyllama_id)

@app.get("/")
async def root():
    return "Hello World!"


@app.post('/Gemini/check')
async def gemini(request: Request):
    try:
        data = await request.json()
        message = data.get("message")
        if message is None:
            raise ValueError("Message field is missing")
        else:
            google_api_key = os.environ['GEMINI_API_KEY']

            genai.configure(api_key=google_api_key)

            model = genai.GenerativeModel('gemini-1.0-pro')

            response = model.generate_content(message)

            text = ""
            for chunk in response:
                text += chunk.text
                text_split = text.split("\n")

            return {"message": text_split}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post('/TinyLlama/check')
async def tinyllama(request: Request):
    try:
        data = await request.json()
        message = data.get("message")
        if message is None:
            raise ValueError("Message field is missing")
        else:
            pipe = pipeline("text-generation",
                            model=model_tiny_llama,
                            tokenizer=tokenizer_tiny_llama,
                            torch_dtype=torch.bfloat16,
                            device_map="auto")

            messages = [
                {"role": "user", "content": f"{message}"},
            ]

            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(prompt, max_new_tokens=300,
                           do_sample=True,
                           temperature=0.7,
                           top_k=50,
                           top_p=0.95)
            results_text = Markdown(outputs[0]["generated_text"])
            return {"message": results_text}
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post('/Phi2/check')
# async def phi2(request: Request):
#     try:
#         data = await request.json()
#         message = data.get("message")
#         if message is None:
#             raise ValueError("Message field is missing")
#         else:
#             pipe = pipeline("text-generation",
#                             model=model_phi,
#                             tokenizer=tokenizer_phi,
#                             trust_remote_code=True,
#                             device_map="auto")
#
#             outputs = pipe(
#                 message,
#                 max_new_tokens=300,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_k=50,
#                 top_p=0.95,
#             )
#             results_text = Markdown(outputs[0]["generated_text"])
#             return {"message": results_text}
#
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))