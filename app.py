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

# Terminal-style HTML interface
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Terminal</title>
    <style>
        body { background: black; color: #00FF00; font-family: monospace; }
        #terminal { width: 80%; margin: auto; }
        textarea, input, button { background: black; color: #00FF00; border: 1px solid #00FF00; font-family: monospace; }
        textarea { width: 100%; height: 100px; }
        input[type=file] { color: white; }
    </style>
</head>
<body>
    <div id="terminal">
        <h2>AI Terminal Interface</h2>
        <form id="aiForm" enctype="multipart/form-data">
            <textarea name="prompt" placeholder="Type your command here..."></textarea><br>
            <input type="file" name="file"><br>
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
            </select><br>
            <button type="submit">Send</button>
        </form>
        <pre id="output"></pre>
    </div>
    <script>
        const form = document.getElementById('aiForm');
        const output = document.getElementById('output');
        form.onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            const res = await fetch('/process', { method: 'POST', body: formData });
            const data = await res.json();
            output.textContent += "\\n> " + formData.get('prompt') + "\\n" + JSON.stringify(data.result, null, 2) + "\\n";
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
    # Call Hugging Face API
    try:
        response = client.chat.completions.create(
            model=model_choice,
            messages=[{"role": "user", "content": prompt}],
            # For image models, you may need additional parameters
        )
    except Exception as e:
        return {"error": str(e)}

    return {"result": response}

# Run the app on fixed port 8080 for Railway free tier
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
