
const form = document.getElementById('upload-form');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', function(event) {
  event.preventDefault(); // Prevent default form submission

  const file = document.getElementById('image-upload').files[0];

  if (!file) {
    alert('Please select an image file.');
    return;
  }

  // Create a FormData object to send the image data
  const formData = new FormData();
  formData.append('file', file);

  // Send an AJAX request to the server using fetch API
  fetch('/predict', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      resultDiv.textContent = data.error;
    } else {
      resultDiv.textContent = `Predicted value: ${data.prediction}`;
    }
  })
  .catch(error => {
    console.error('Error sending image data:', error);
    resultDiv.textContent = 'An error occurred. Please try again.';
  });
});
