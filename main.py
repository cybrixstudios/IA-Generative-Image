from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import base64
import tempfile

app = Flask(__name__)

# Cargar modelo una sola vez al iniciar
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe = pipe.to(device)

@app.route("/", methods=["GET"])
def home():
    return "Servidor Cybrix Imagine IA está activo. Usa /generar para enviar prompts."

@app.route("/generar", methods=["POST"])
def generar():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt vacío"}), 400

    try:
        result = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

        # Guardar imagen temporal
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        result.save(temp.name)

        return send_file(temp.name, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
