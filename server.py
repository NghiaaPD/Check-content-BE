from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv
import google.generativeai as genai
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import asyncio
import re
import uvicorn

app = FastAPI()

load_dotenv()

uri = os.environ['MONGODB_URI']
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


@app.get("/")
async def root():
    return "Hello World!"


@app.post('/Gemini/check')
async def gemini(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        answer = data.get("answer")
        if question is None or answer is None:
            raise ValueError("Question or answer field is missing")

        google_api_key = os.environ['GEMINI_API_KEY']
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-1.0-pro')

        db = client['check-content']
        collection = db['SSL101c']

        n = 0
        while True:
            result = collection.find({}, {"Answer": 1, "Question": 1, "_id": 0}).skip(n).limit(1)
            doc = next(result, None)  # Get the next document or None if no more documents
            if doc is None:
                raise HTTPException(status_code=404, detail="No document with Overlap ratio greater than 70% found")

            answer_db = doc.get("Answer")
            question_db = doc.get("Question")

            prompt = f"""
            Please compare for me the overlap between the structure and meaning of that question
            {question} and the answer {answer} with the following question and answer: {question_db} and {answer_db}. 
            Then JUST RETURN to me the: 
            - percentage of similarity
            - structural similarity ratio
            - semantic overlap ratio and 
            - the most duplicated words 
            between the two pairs of question and answer with EXAMPLE of the output format as below:

            Question need checking:
            question:{question}
            answer:{answer}

            Question in database:
            question:{question_db}
            answer:{answer_db}

            Overlap ratio: 0%
            Structural similarity ratio: 0%
            Semantic overlap ratio: 0%
            Most duplicated words: None
            """

            response = model.generate_content(prompt)

            text = "".join(chunk.text for chunk in response)

            # Extract the Overlap ratio from the response text
            lines = text.split("\n")
            overlap_ratio = 0
            for line in lines:
                if "Overlap ratio:" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        overlap_ratio = float(match.group(1))
                        break

            if overlap_ratio > 70:
                return {"message": text}

            n += 1  # Increment the document index
            await asyncio.sleep(5)  # Delay for 5 seconds before the next request

    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {err}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)