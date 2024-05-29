from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv
import google.generativeai as genai
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

app = FastAPI()

load_dotenv()


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
            pass
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
