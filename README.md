# Non-invasive detection of apple bruises with ANN and thermal image processing
# Overview
This project presents a non-invasive approach for detecting bruises in apples using thermal imaging and Artificial Neural Networks (ANNs). The method enhances fruit quality assessment by automating bruise detection, reducing manual errors, and minimizing food waste.
Detecting bruises in apples is crucial for maintaining their quality.Current methods, like visual inspection, can miss internal bruises. This study aims to explore the use of  thermal imaging for non-destructive bruise detection in apples. The goal is to develop a reliable method to identify bruises accurately,improving quality control and reducing food waste.
# Features
-Thermal Imaging: Captures temperature variations to detect bruises.
-K-means Clustering: Segments images for bruise classification.
-Feature Extraction: Analyzes temperature distribution, texture, and color differences.
-Artificial Neural Network (ANN): Classifies bruised and non-bruised apples.
-K-Fold Cross Validation: Ensures model reliability and generalization.
# Methodology
-Image Capture: Thermal images of apples are collected.
-Data Augmentation: Enhances dataset with variations for better model training.
-K-means Clustering: Segments bruised and non-bruised areas.
-Feature Extraction: Identifies patterns in bruised regions.
-Training ANN: The model learns from labeled data.
-Classification & Validation: Predicts bruises and evaluates accuracy.

![WhatsApp Image 2025-01-31 at 11 50 59 AM](https://github.com/user-attachments/assets/b85dce18-8e8a-4390-8b58-4d334fd3250e)

# Output
Classification of Bruised and Non-Bruised Finally the trained model detects whether or not there are bruised images on thermal images. By identifying if fruit has bruises the ANN'S output allows one to differentiate between healthy and damaged fruits. By ensuring that only unbruised fruit is stored for long time by ensuring that only unbruised fruit is kept for storage in the freezer for 7 months. This information can subsequently be applied to reduce fruit waste in storage over extended periods of time.

![image](https://github.com/user-attachments/assets/c4dbb126-10af-44ce-81f5-94440b848aba)

# Results
-Precision: 89.28%
-Recall: 93.18%
-F1 Score: 91.12%
-Comparison: Thermal imaging outperforms standard webcam images in detecting internal bruises.

![WhatsApp Image 2025-01-30 at 9 29 32 AM](https://github.com/user-attachments/assets/c51695b7-0b58-4978-95a7-c7a5877fd4c3)

The confusion matrix and its associated metrics, such as accuracy, precision, recall, 
and F1 score, are crucial for evaluating and understanding the performance of a classification model. 
They ensure a trustworthy and accurate model by offering a comprehensive analysis of projections and 
highlighting potential areas for improvement.

# Conclusion
This project developed an automated system for detecting apple bruises using thermal imaging and Artificial Neural Networks (ANN). The method effectively identifies bruised areas that are invisible to the naked eye, improving accuracy over traditional manual inspections. By leveraging computer vision and machine learning, the system enhances efficiency, reduces labor, and ensures better fruit quality. Future improvements could extend this approach to other fruits and integrate a web-based interface for real-time analysis.


