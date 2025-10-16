import os
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from openai import OpenAI
import uvicorn

app = FastAPI()

# Initialize OpenAI client (Hugging Face API)
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)

# Mobile-friendly chat interface with loader
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Chat App</title>
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
        #loader { text-align: center; padding: 10px; display: none; font-style: italic; color: #555; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div id="messages"></div>
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

        function addMessage(content, sender) {
            const div = document.createElement('div');
            div.className = 'message ' + sender;
            div.textContent = content;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }

        form.onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            const userText = formData.get('prompt');
            if(!userText) return;

            addMessage(userText, 'user');
            loader.style.display = 'block';

            try {
                const res = await fetch('/process', { method: 'POST', body: formData });
                const data = await res.json();
                addMessage(data.result, 'ai');
            } catch (err) {
                addMessage('Error: ' + err.message, 'ai');
            } finally {
                loader.style.display = 'none';
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
    # Read image if uploaded
    image_bytes = await file.read() if file else None

    try:
        response = client.chat.completions.create(
            model=model_choice,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract the AI message
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Error: {str(e)}"

    return {"result": reply}

# Run on fixed port 8080 for Railway free tier
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
