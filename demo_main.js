const API_URL = window.location.origin;
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');

// Click to upload
dropZone.addEventListener('click', () => fileInput.click());

// Drag and drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

async function handleFile(file) {
    // Validate file
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload JPG, PNG, or WebP.');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
        showError('File too large. Maximum size is 10MB.');
        return;
    }
    
    // Show loading
    uploadSection.style.display = 'none';
    loadingSection.style.display = 'block';
    
    // Create form data
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch(`${API_URL}/api/analyze`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    }
}

function displayResults(data) {
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Update face shape
    document.getElementById('primaryShape').textContent = data.face_shape.primary;
    document.getElementById('confidence').textContent = 
        `${(data.face_shape.confidence * 100).toFixed(1)}% confidence`;
    
    // Update all shapes bar chart
    const allShapesContainer = document.getElementById('allShapes');
    allShapesContainer.innerHTML = '';
    
    Object.entries(data.face_shape.all_probabilities)
        .sort((a, b) => b[1] - a[1])
        .forEach(([shape, prob]) => {
            const bar = document.createElement('div');
            bar.className = 'shape-bar';
            bar.innerHTML = `
                <div class="shape-bar-label">
                    <span>${shape}</span>
                    <span>${(prob * 100).toFixed(1)}%</span>
                </div>
                <div class="shape-bar-fill">
                    <div class="shape-bar-progress" style="width: 0%"></div>
                </div>
            `;
            allShapesContainer.appendChild(bar);
            
            // Animate bar
            setTimeout(() => {
                bar.querySelector('.shape-bar-progress').style.width = `${prob * 100}%`;
            }, 100);
        });
    
    // Update image
    document.getElementById('uploadedImage').src = `${API_URL}${data.image_url}`;
    
    // Update recommendations
    const recContainer = document.getElementById('recommendedStyles');
    recContainer.innerHTML = '';
    data.recommendations.recommended.forEach((style, index) => {
        const item = document.createElement('div');
        item.className = 'style-item';
        item.style.animationDelay = `${index * 0.1}s`;
        item.textContent = style;
        recContainer.appendChild(item);
    });
    
    // Update avoid list
    const avoidContainer = document.getElementById('avoidStyles');
    avoidContainer.innerHTML = '';
    data.recommendations.avoid.forEach((style, index) => {
        const item = document.createElement('div');
        item.className = 'style-item';
        item.style.animationDelay = `${index * 0.1}s`;
        item.textContent = style;
        avoidContainer.appendChild(item);
    });
    
    // Update reasoning
    const reasoningContainer = document.getElementById('reasoningText');
    reasoningContainer.innerHTML = '';
    data.recommendations.reasoning.forEach(reason => {
        const p = document.createElement('p');
        p.textContent = reason;
        reasoningContainer.appendChild(p);
    });
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    loadingSection.style.display = 'none';
    errorSection.style.display = 'block';
    document.getElementById('errorText').textContent = message;
}

function resetApp() {
    uploadSection.style.display = 'block';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    fileInput.value = '';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Check API health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/api/health`);
        const data = await response.json();
        if (!data.model_loaded) {
            console.warn('Model not loaded on server');
        }
    } catch (error) {
        console.error('API health check failed:', error);
    }
});
