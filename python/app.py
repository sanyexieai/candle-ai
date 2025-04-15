from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import os
import tempfile
from predict_mnist import load_model, predict
import time

app = Flask(__name__)

# Load the model
try:
    # Get the absolute path to the model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "mnist_model_best.safetensors")
    print(f"Loading model from: {model_path}")
    model, device = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    temp_file_path = None
    try:
        start = time.time()
        
        # Get the base64 encoded image data
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        image_data = data['image'].split(',')[1]
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file_path = temp_file.name
        temp_file.close()  # Close the file handle immediately
        
        try:
            # Convert base64 to image and save
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            image.save(temp_file_path)
            image.close()  # Close the image handle
            
            # Use predict_mnist's predict function
            result = predict(temp_file_path, model, device)
            
            return jsonify(result)
        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file: {str(e)}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 