from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import base64
import asyncio
import time
import io
import uvicorn

app = FastAPI()

# Load model and processor
MODEL_ID = "Jise/qwen2.5-7b-instruct-citynav"
DEVICE_COUNT = torch.cuda.device_count()
models = []
processors = []
for i in range(DEVICE_COUNT):
    models.append(Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": i}))
    processors.append(Qwen2_5_VLProcessor.from_pretrained(MODEL_ID))
MAX_BATCH_SIZE = 4
MAX_WAIT_TIME = 0.05

class ContentItem(BaseModel):
    type: str
    text: str = None
    image_url: dict = None  # expects {"url": "data:image/...base64,..."}

class ChatMessage(BaseModel):
    role: str
    content: list[ContentItem]

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256

class RequestItem:
    def __init__(self, messages, future, max_tokens):
        self.messages = messages
        self.future = future
        self.max_tokens = max_tokens
        
request_queues = [asyncio.Queue() for _ in range(DEVICE_COUNT)]

def decode_base64_image(base64_str):
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(base64_str)))

async def batching_worker(device_id):
    model = models[device_id]
    processor = processors[device_id]
    request_queue = request_queues[device_id]
    device = torch.device(f"cuda:{device_id}")
    while True:
        batch = []
        futures = []
        max_tokens = []
        start = time.time()
        
        while len(batch) < MAX_BATCH_SIZE and (time.time() - start) < MAX_WAIT_TIME:
            try:
                item = request_queue.get_nowait()
                batch.append(item.messages)
                futures.append(item.future)
                max_tokens.append(item.max_tokens)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.005)
            
            if not batch:
                continue
            
            all_text_inputs = []
            all_images = []
            
            for msg_seq in batch:
                context = []
                image_inputs = []
                for msg in msg_seq:
                    entry = {"role": msg.role, "content": []}
                    for item in msg.content:
                        if item.type == "text":
                            entry["content"].append({"type": "text", "text": item.text})
                        elif item.type == "image_url":
                            image_inputs.append(decode_base64_image(item.image_url["url"]))
                            entry["content"].append({"type": "image", "image": image_inputs[-1]})
                    context.append(entry)
                
                text_input = processor.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
                all_text_inputs.append(text_input)
                all_images.append(image_inputs)
            
            inputs = processor(text=all_text_inputs, images=all_images, return_tensors="pt", padding=True).to(device)
            output_ids = model.generate(**inputs, max_new_tokens=max(max_tokens))
            prompt_lengths = inputs.attention_mask.sum(dim=1).tolist()
            trimmed_generated_ids = [out_ids[prompt_len:] for prompt_len, out_ids in zip(prompt_lengths, output_ids)]
            decoded = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for future, decoded_text in zip(futures, decoded):
                future.set_result(decoded_text)

round_robin_counter = 0

def get_next_device():
    global round_robin_counter
    device_id = round_robin_counter % DEVICE_COUNT
    round_robin_counter += 1
    return device_id
            
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    future = asyncio.get_event_loop().create_future()
    item = RequestItem(messages=request.messages, future=future, max_tokens=request.max_tokens)
    device_id = get_next_device()
    await request_queues[device_id].put(item)
    result = await future
    
    return {
        "id": "chatcmpl-generated",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}],
        "model": request.model
    }

@app.on_event("startup")
async def startup_event():
    for i in range(DEVICE_COUNT):
        asyncio.create_task(batching_worker(i))
    print("Batching workers started.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")