import asyncio
import json
from fastapi.responses import StreamingResponse
from typing import List, Optional, Union, AsyncGenerator
import time
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException
import signal
import sys

# Environment variable for Hathora region
HATHORA_REGION = os.environ.get("HATHORA_REGION", "unknown")

# --- Model/Device Setup ---


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_name: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer


MODEL_NAME = os.environ.get("MODEL_NAME", "distilbert/distilgpt2")
print(f"Loading model: {MODEL_NAME}")
DEVICE = get_device()
MODEL, TOKENIZER = load_model_and_tokenizer(MODEL_NAME, DEVICE)

# --- Pydantic Models for OpenAI compatibility ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 50


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    hathora_region: str = HATHORA_REGION
    time_to_first_token: Optional[float] = None
    total_inference_latency: float
    device: str

# --- Utility Functions ---


def extract_prompt(messages: List[ChatMessage]) -> str:
    last_user_message = next(
        (msg.content for msg in reversed(messages) if msg.role == 'user'), None)
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    return last_user_message


def tokenize_prompt(tokenizer, prompt: str, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def generate_response(
    model, tokenizer, inputs, prompt, max_tokens: int
):
    inference_start_time = time.time()
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=inputs["input_ids"].shape[1] + max_tokens,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            top_k=50,
        )
    inference_end_time = time.time()
    total_inference_latency_ms = (
        inference_end_time - inference_start_time) * 1000
    generated_text = tokenizer.decode(
        output_sequences[0], skip_special_tokens=True)
    response_text = generated_text[len(prompt):].strip()
    return response_text, output_sequences, total_inference_latency_ms


def build_non_streaming_response(
    request: ChatCompletionRequest,
    response_text: str,
    output_sequences,
    inputs,
    total_inference_latency_ms: float,
    device: str
) -> ChatCompletionResponse:
    completion_message = ChatMessage(role="assistant", content=response_text)
    choice = ChatCompletionChoice(index=0, message=completion_message)
    usage = UsageInfo(
        prompt_tokens=len(inputs["input_ids"][0]),
        completion_tokens=len(
            output_sequences[0]) - len(inputs["input_ids"][0]),
        total_tokens=len(output_sequences[0])
    )
    return ChatCompletionResponse(
        id=f"cmpl-{''.join(str(time.time()).split('.'))}",
        model=request.model,
        choices=[choice],
        usage=usage,
        total_inference_latency=total_inference_latency_ms,
        device=device,
    )


async def stream_tokens(
    model, tokenizer, inputs, prompt, max_tokens: int, device: str
) -> AsyncGenerator[str, None]:
    input_ids = inputs["input_ids"]
    generated = input_ids
    past_key_values = None
    start_time = time.time()
    for i in range(max_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated if i == 0 else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token_id], dim=-1)
        token = tokenizer.decode(next_token_id[0])
        time_to_token = (time.time() - start_time) * 1000  # ms
        data = {
            "id": f"cmpl-{''.join(str(time.time()).split('.'))}",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": {"content": token},
                    "index": 0,
                    "finish_reason": None if i < max_tokens - 1 else "length"
                }
            ],
            "hathora_region": HATHORA_REGION,
            "device": device,
            "time_to_token": time_to_token
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0)  # Yield control to event loop
    yield "data: [DONE]\n\n"
    await asyncio.sleep(0)

# --- FastAPI App and Endpoints ---
app = FastAPI()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, stream: Optional[bool] = False):
    prompt = extract_prompt(request.messages)
    inputs = tokenize_prompt(TOKENIZER, prompt, DEVICE)
    device_str = str(DEVICE)
    if not stream:
        response_text, output_sequences, total_inference_latency_ms = generate_response(
            MODEL, TOKENIZER, inputs, prompt, request.max_tokens
        )
        return build_non_streaming_response(
            request, response_text, output_sequences, inputs, total_inference_latency_ms, device_str
        )
    else:
        return StreamingResponse(
            stream_tokens(MODEL, TOKENIZER, inputs, prompt,
                          request.max_tokens, device_str),
            media_type="text/event-stream"
        )


@app.get("/health")
async def health_check():
    return {"status": "ok"}
