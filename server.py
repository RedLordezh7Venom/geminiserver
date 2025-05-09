import os
import google.generativeai as genai
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise KeyError
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set it before running the application.")
    print("Example: export GOOGLE_API_KEY='YOUR_API_KEY_HERE'")
    exit(1)

# --- FastAPI App Setup ---
app = FastAPI()

# --- Gemini Model Setup ---
MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_NAME)

# --- Request Model ---
class PromptRequest(BaseModel):
    prompt: str

# --- API Endpoint ---
@app.post("/gemini")
async def query_gemini(request: PromptRequest):
    prompt_text = request.prompt
    if not prompt_text:
        raise HTTPException(status_code=400, detail="Missing 'prompt' in JSON payload")

    try:
        response = model.generate_content(prompt_text)
        if hasattr(response, "parts") and response.parts:
            result_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text'):
            result_text = response.text
        else:
            result_text = "No text content found in response."
            print(f"Full Gemini Response: {response}")

        return {"response": result_text}
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

# --- Run with: uvicorn server:app --host 0.0.0.0 --port 5000 --reload ---