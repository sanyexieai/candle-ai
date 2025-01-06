import init, { ImagePredictor } from './pkg/rust_wasm_example.js';

let predictor = null;

async function loadModel(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
        await init();
        const arrayBuffer = await file.arrayBuffer();
        const weights = new Uint8Array(arrayBuffer);
        predictor = new ImagePredictor(weights);
        
        // 启用图片选择
        document.getElementById('image-input').disabled = false;
        document.getElementById('model-input').disabled = true;
        
        document.getElementById('prediction-text').textContent = '模型加载成功，请选择图片';
    } catch (error) {
        console.error('加载模型失败:', error);
        document.getElementById('prediction-text').textContent = '加载模型失败: ' + error.message;
    }
}

async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const preview = document.getElementById('preview');
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
    
    document.getElementById('predict-btn').disabled = false;
}

async function predictImage() {
    if (!predictor) {
        alert('请先加载模型');
        return;
    }

    const file = document.getElementById('image-input').files[0];
    if (!file) return;

    const loading = document.getElementById('loading');
    const predictBtn = document.getElementById('predict-btn');
    
    loading.style.display = 'block';
    predictBtn.disabled = true;

    try {
        const arrayBuffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        
        const result = await predictor.predict(uint8Array);
        const [prediction, timing] = result.split('\n');
        
        document.getElementById('prediction-text').textContent = prediction;
        document.getElementById('timing').textContent = timing;
    } catch (error) {
        console.error('预测出错:', error);
        document.getElementById('prediction-text').textContent = '预测失败: ' + error.message;
    } finally {
        loading.style.display = 'none';
        predictBtn.disabled = false;
    }
}

// 初始化事件监听
document.getElementById('model-input').addEventListener('change', loadModel);
document.getElementById('image-input').addEventListener('change', handleImageUpload);
document.getElementById('predict-btn').addEventListener('click', predictImage);