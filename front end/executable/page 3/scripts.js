
document.getElementById('tripForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const destination = document.getElementById('destination').value;
    const date = document.getElementById('date').value;

    const tripDetails = `
        <h2>Your Trip Details</h2>
        <p><strong>Destination:</strong> ${destination}</p>
        <p><strong>Date:</strong> ${date}</p>
    `;

    document.getElementById('tripDetails').innerHTML = tripDetails;
});
