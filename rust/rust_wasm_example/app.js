import init, { ConvNetModel } from './build/m.js';

async function loadAndPredict() {
    const weightsInput = document.getElementById('weights-input');
    const imageInput = document.getElementById('image-input');
    const resultDisplay = document.getElementById('prediction-result');

    if (weightsInput.files.length === 0 || imageInput.files.length === 0) {
        resultDisplay.textContent = 'Please select both a weights file and an image file.';
        return;
    }

    const weightsFile = weightsInput.files[0];
    const imageFile = imageInput.files[0];

    // Initialize the WebAssembly module
    await init();

    // Load weights
    const weightsArrayBuffer = await weightsFile.arrayBuffer();
    const model = new ConvNetModel(new Uint8Array(weightsArrayBuffer));

    // Load and predict image
    const imageArrayBuffer = await imageFile.arrayBuffer();
    const prediction = await model.predict_image(new Uint8Array(imageArrayBuffer), 28, 28);
    resultDisplay.textContent = 'Prediction: ' + prediction;
}

document.addEventListener('DOMContentLoaded', () => {
  // Ensure the page is fully loaded before attaching the event handler
  document.querySelector('button').addEventListener('click', loadAndPredict);
});

// Ensure wasm-pack
