from flask import Flask, request, render_template
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
import torch
import os
import traceback

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

app = Flask(__name__)

# Загрузите модель и токенизатор
model_name = "/models/Qwen/Qwen2-VL-7B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
device = torch.device("cuda")
model.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def process():
    try:
        text = request.form['text']
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, 
            max_length=1200,
            num_beams=5,
            temperature=0.3,
            top_p=0.85, 
            num_return_sequences=1)
        
        
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {'msg': f"Result: {result_text}"}
    except Exception as e:
        return {'msg': f"An error occurred: {traceback.format_exc()}"}

if __name__ == '__main__':
    app.run(debug=True)
