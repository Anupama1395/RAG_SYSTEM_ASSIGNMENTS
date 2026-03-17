import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()

client = OpenAI()  # reads OPENAI_API_KEY from environment

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1,
strip_whitespace=True)

@app.get("/")
def read_root():
    return {"message": "LLM API is running"}

@app.post("/hello")
def hello(request: PromptRequest):
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=request.prompt
        )

        llm_output = response.output[0].content[0].text

        return {"input": request.prompt, "llm_output": llm_output}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again later.")