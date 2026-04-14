"""Launch the FastAPI backend."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uvicorn
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )
