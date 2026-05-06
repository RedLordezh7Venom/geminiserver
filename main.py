import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import PlainTextResponse
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# The Groq client automatically picks up GROQ_API_KEY from the environment,
# but we can also explicitly fetch it using os.getenv as requested
if os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Groq LLM Fast API", description="A fast REST API to question the Groq LLM via GET requests.")

# Initialize the async Groq client for better performance in FastAPI
client = AsyncGroq()

@app.get("/ask", response_class=PlainTextResponse)
async def ask_question(q: str = Query(..., description="The question to ask the LLM")):
    try:
        # Using Llama 3 8B model for extremely fast responses
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": q,
                }
            ],
            model="openai/gpt-oss-120b",
            temperature=0.4,
            max_tokens=6192,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Groq API is running. Use GET /ask?q=your_question to ask a question."}

# --- Run with Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    # Render assigns the PORT environment variable dynamically, but we default to 5000
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
