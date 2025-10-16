import os
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn
import base64

app = FastAPI()

# Enable CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hugging Face client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)

# HTML content for mobile-friendly chat
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Vida AI Chat</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { margin:0; padding:0; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background: #f0f0f0; }
        .chat-container { max-width: 480px; margin: 0 auto; height: 100vh; display: flex; flex-direction: column; background: #fff; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); overflow: hidden; }
        #messages { flex: 1; padding: 10px; overflow-y: auto; display: flex; flex-direction: column; }
        .message { padding: 10px 15px; margin: 5px 0; border-radius: 20px; max-width: 80%; word-wrap: break-word; }
        .user { background: #007AFF; color: #fff; align-self: flex-end; border-bottom-right-radius: 0; }
        .ai { background: #e5e5ea; color: #000; align-self: flex-start; border-bottom-left-radius: 0; }
        form { display: flex; padding: 10px; border-top: 1px solid #ccc; background: #fafafa; flex-wrap: wrap; }
        input[type=text] { flex: 1; padding: 10px; border-radius: 20px; border: 1px solid #ccc; margin-right: 5px; }
        select { margin: 5px 0; border-radius: 10px; padding: 5px; flex-basis: 100%; }
        input[type=file] { flex-basis: 100%; margin: 5px 0; }
        button { padding: 10px 20px; border-radius: 20px; border: none; background: #007AFF; color: #fff; cursor: pointer; }
        #loader { width: 100%; text-align: center; padding: 5px; display: none; font-style: italic; color: #555; }
        #progress-bar { width: 100%; height: 5px; background: #e5e5ea; display: none; margin-bottom: 5px; border-radius: 5px; }
        #progress-fill { width: 0%; height: 100%; background: #007AFF; border-radius: 5px; }
        img.ai-image { max-width: 100%; border-radius: 10px; margin: 5px 0; }
    </style>
</head>
<body>
<div class="chat-container">
    <div id="messages"></div>
    <div id="progress-bar"><div id="progress-fill"></div></div>
    <div id="loader">AI is typing...</div>
    <form id="aiForm" enctype="multipart/form-data">
        <input type="text" name="prompt" placeholder="Type your message..." required>
        <input type="file" name="file">
        <select name="model_choice">
            <option value="inclusionAI/Ling-1T">Ling-1T (Text)</option>
            <option value="deepseek-ai/DeepSeek-R1">DeepSeek-R1 (Image)</option>
            <option value="microsoft/UserLM-8b">UserLM-8b</option>
            <option value="zai-org/GLM-4.6">GLM-4.6</option>
            <option value="LiquidAI/LFM2-8B-A1B">LFM2-8B-A1B</option>
            <option value="Qwen/Qwen3-8B">Qwen3-8B</option>
            <option value="google/flan-t5-large">Flan-T5-Large</option>
            <option value="Phr00t/Qwen-Image-Edit-Rapid-AIO">Qwen-Image-Edit-Rapid-AIO</option>
            <option value="tencent/HunyuanImage-3.0">HunyuanImage-3.0</option>
            <option value="black-forest-labs/FLUX.1-dev">FLUX.1-dev</option>
        </select>
        <button type="submit">Send</button>
    </form>
</div>

<script>
const form = document.getElementById('aiForm');
const messages = document.getElementById('messages');
const loader = document.getElementById('loader');
const progressBar = document.getElementById('progress-bar');
const progressFill = document.getElementById('progress-fill');

function addMessage(content, sender, isImage=false) {
    const div = document.createElement('div');
    div.className = 'message ' + sender;
    if (isImage) {
        const img = document.createElement('img');
        img.src = content;
        img.className = 'ai-image';
        div.appendChild(img);
    } else {
        div.textContent = content;
    }
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

async function simulateProgress() {
    progressBar.style.display = 'block';
    progressFill.style.width = '0%';
    let width = 0;
    return new Promise(resolve => {
        const interval = setInterval(() => {
            width += Math.random() * 10;
            if(width >= 90) { width = 90; clearInterval(interval); resolve(); }
            progressFill.style.width = width + '%';
        }, 200);
    });
}

form.onsubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const userText = formData.get('prompt');
    if(!userText) return;

    addMessage(userText, 'user');
    loader.style.display = 'block';
    await simulateProgress();

    try {
        const res = await fetch('/process', { method: 'POST', body: formData });
        const data = await res.json();
        loader.style.display = 'none';
        progressFill.style.width = '100%';

        // Detect if response is image (base64)
        if(data.result.startsWith('data:image')) {
            addMessage(data.result, 'ai', true);
        } else {
            addMessage(data.result, 'ai');
        }
    } catch (err) {
        loader.style.display = 'none';
        addMessage('Error: Failed to fetch', 'ai');
    } finally {
        progressBar.style.display = 'none';
        form.reset();
    }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return html_content

@app.post("/process")
async def process(
    prompt: str = Form(...),
    file: UploadFile | None = File(None),
    model_choice: str = Form(...)
):
    image_bytes = await file.read() if file else None

    try:
        response = client.chat.completions.create(
            model=model_choice,
            messages=[{"role": "user", "content": prompt}],
        )
        # Safely extract AI reply
        reply = getattr(response.choices[0].message, 'content', str(response))
    except Exception as e:
        reply = f"Error: {str(e)}"

    # Optional: if image model, convert bytes to base64 (if your model returns images)
    if image_bytes:
        encoded = base64.b64encode(image_bytes).decode()
        reply = f"data:image/png;base64,{encoded}"

    return {"result": reply}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
