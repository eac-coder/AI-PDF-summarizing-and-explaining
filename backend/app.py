from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Hugging Face model URL (we'll use a free instruction-following model)
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

@app.route('/process', methods=['POST'])
def process_pdf():
    text = request.json.get("text")
    option = request.json.get("option")  # "Summarize", "Explain", "Both"

    if not text or not option:
        return jsonify({"error": "Missing text or option"}), 400

    # Prepare the prompt based on the option
    if option == "Summarize":
        prompt = f"Summarize this text clearly and concisely:\n{text}"
    elif option == "Explain":
        prompt = f"Explain this text step by step for a beginner:\n{text}"
    elif option == "Both":
        prompt = f"Summarize and then explain this text for a beginner:\n{text}"
    else:
        prompt = text  # fallback

    # Send to Hugging Face model
    response = requests.post(API_URL, json={"inputs": prompt})
    try:
        result = response.json()
        # Hugging Face API may return a list of dicts
        output_text = result[0]["generated_text"] if isinstance(result, list) else str(result)
    except Exception:
        output_text = "Error: Could not process text via Hugging Face."

    return jsonify({"result": output_text})

if __name__ == "__main__":
    app.run()
