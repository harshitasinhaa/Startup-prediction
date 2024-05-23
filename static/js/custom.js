document.getElementById('predictionForm').onsubmit = function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    
    fetch('{% url 'mmlmodel:predict' %}', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('predictionForm').style.display = 'none';
        const resultDiv = document.getElementById('predictionResult');
        resultDiv.querySelector('.prediction-text').textContent = data.prediction || 'No prediction available';
        resultDiv.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        const resultDiv = document.getElementById('predictionResult');
        resultDiv.querySelector('.prediction-text').textContent = 'Failed to predict, please try again.';
        resultDiv.style.display = 'block';
    });
};

function startOver() {
    document.getElementById('predictionResult').style.display = 'none';
    document.getElementById('predictionForm').style.display = 'block';
}
