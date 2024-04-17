function searchLocation() {
    const searchInput = document.getElementById('searchInput');
    const predictionResult = document.getElementById('predictionResult');
    const location = searchInput.value;

    // Fetch prediction data from backend (replace with your backend endpoint)
    fetch(`/predict?location=${location}`)
        .then(response => response.json())
        .then(data => {
            predictionResult.innerHTML = `Predicted crowd density for ${location}: ${data.prediction}`;
        })
        .catch(error => {
            console.error('Error fetching prediction data:', error);
            predictionResult.innerHTML = 'Error fetching prediction data.';
        });
}
