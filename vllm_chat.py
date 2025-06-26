from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import RedirectResponse
from vllm import LLM, SamplingParams
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app and router
app = FastAPI()
vllm_router = APIRouter()

# Load the LLM model
llm = LLM(
    model="/home/z890/model/Meta-Llama-3-8B",
    # tensor_parallel_size=2,
    dtype="float16"
)

# Define request schema
class PromptRequest(BaseModel):
    prompts: list[str]
    temperature: float
    top_p: float
    max_tokens: int

# Root endpoint to redirect to API docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# Endpoint for generating responses using vLLM
@vllm_router.post("/vllm_chat")
async def vllm_chat(request: PromptRequest): #prompts: list[str], temperature: float, top_p: float, max_tokens: int
    """Generate text responses from vLLM"""
    try:  
        sampling_params = SamplingParams(temperature=request.temperature, top_p=request.top_p, max_tokens=request.max_tokens)
        outputs = llm.generate(request.prompts, sampling_params)
        
        print("outputs=====", outputs)
        
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        return {"responses": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for stopping chat (implementation placeholder)
@vllm_router.post("/vllm_chat_stop")
async def stop_chat(device_name: str, password: str):
    """Stop the LLM chat process"""
    try:
        # Placeholder for stopping logic
        return {"message": f"Chat process for {device_name} stopped."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the router
app.include_router(vllm_router, prefix="/vllm")

# Run the application
if __name__ == "__main__":
    print("API-Version: 2024-10-08")
    uvicorn.run(app, host="127.0.0.1", port=8070)