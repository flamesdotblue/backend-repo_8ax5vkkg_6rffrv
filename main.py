import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    context: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    provider: str
    fallback: bool = False


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        # Try to import database module
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


def _fallback_answer(prompt: str) -> str:
    guidance = (
        "I'm an on-site fitness assistant. I can help with workouts, exercise form, programs, and nutrition. "
        "Since no AI provider key is configured, this is a smart fallback summary based on your question.\n\n"
    )
    safety = (
        "Important: This is general information and not medical advice. Consult a professional for personalized guidance."
    )
    return f"{guidance}Your question: {prompt}\n\n{safety}"


@app.post("/ai/chat", response_model=ChatResponse)
async def ai_chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return ChatResponse(
            reply=_fallback_answer(message),
            model="fallback-local",
            provider="local",
            fallback=True,
        )

    # Try OpenAI API
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        system_prompt = (
            "You are FitGuide AI, a friendly, evidence-based fitness and nutrition assistant. "
            "Be concise, practical, and safe. Use bullet points when helpful. "
            "If health risks are present, recommend consulting a professional."
        )

        messages = [{"role": "system", "content": system_prompt}]
        # include history if provided
        if req.history:
            for m in req.history:
                messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": message})

        # Use a small, fast model to keep latency low if available
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.4,
            max_tokens=600,
        )
        reply = completion.choices[0].message.content or ""
        return ChatResponse(
            reply=reply.strip(),
            model=completion.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            provider="openai",
            fallback=False,
        )
    except Exception as e:
        # graceful fallback if provider fails
        return ChatResponse(
            reply=_fallback_answer(message) + f"\n\n(Note: Provider error: {str(e)[:120]})",
            model="fallback-local",
            provider="local",
            fallback=True,
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
