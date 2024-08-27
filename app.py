from flask import Flask, render_template, request, url_for
from PIL import Image
import numpy as np
import pickle
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
import base64
from io import BytesIO

app = Flask(__name__)

# Load the scaler
scaler = pickle.load(open('scaler1.pkl', 'rb'))

def extract_features(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Extract GLCM features
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    features = {
        'Contrast': graycoprops(glcm, 'contrast')[0, 0],
        'Correlation': graycoprops(glcm, 'correlation')[0, 0],
        'Energy': graycoprops(glcm, 'energy')[0, 0],
        'Homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'Mean': np.mean(gray_image),
        'Standard Deviation': np.std(gray_image),
        'Entropy': -np.sum(glcm * np.log2(glcm + (glcm == 0))),
        'Variance': np.var(gray_image),
        'Skewness': np.mean(((gray_image - np.mean(gray_image)) / np.std(gray_image)) ** 3),
        'Kurtosis': np.mean(((gray_image - np.mean(gray_image)) / np.std(gray_image)) ** 4) - 3
    }
    return features

def process_and_classify_image(image):
    # Extract features
    features = extract_features(image)
    
    # Prepare feature vector
    feature_vector = np.array([[
        features['Contrast'], features['Correlation'], features['Energy'],
        features['Homogeneity'], features['Mean'], features['Standard Deviation'],
        features['Entropy'], features['Variance'], features['Kurtosis'],
        features['Skewness']
    ]])
    
    # Scale features
    scaled_features = scaler.transform(feature_vector)
    
    # Mock prediction logic (since TensorFlow is removed)
    bruised_prob = np.random.rand()
    non_bruised_prob = 1 - bruised_prob
    
    threshold = 0.5
    if bruised_prob >= threshold:
        result = "Prediction: Bruised 🥺"
        color = "red"
    elif 0.3010 <= bruised_prob <= 0.5046:
        result = "Prediction: Moderate Bruised 😕"
        color = "orange"
    else:
        result = "Prediction: Non-Bruised 😊"
        color = "green"
        
    return non_bruised_prob, bruised_prob, result, color

def cluster_image(image, n_clusters=3):
    # Convert the PIL image to a format suitable for clustering
    image = np.array(image)
    
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_values)
    
    # Get the labels and cluster centers
    labels = kmeans.labels_
    centers = np.uint8(kmeans.cluster_centers_)

    # Convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Try PNG instead of JPEG
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part in the request"
        
        uploaded_file = request.files['image']
        
        if uploaded_file.filename == '':
            return "No file chosen"

        if uploaded_file:
            image = Image.open(uploaded_file)
            
            # Generate cluster images
            cluster_images = []
            for n_clusters in range(2, 5):  # Generate clusters for 2, 3, and 4 clusters
                clustered_img = cluster_image(image, n_clusters)
                clustered_img = Image.fromarray(clustered_img)
                base64_img = convert_image_to_base64(clustered_img)
                cluster_images.append(base64_img)
                print(f"Generated cluster image with {n_clusters} clusters")
                print(f"Base64 string length: {len(base64_img)}")

            # Process and classify image
            non_bruised_prob, bruised_prob, result, color = process_and_classify_image(image)

            return render_template('index.html', 
                                   cluster_images=cluster_images,
                                   result=result, 
                                   color=color, 
                                   non_bruised_prob=non_bruised_prob, 
                                   bruised_prob=bruised_prob)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
