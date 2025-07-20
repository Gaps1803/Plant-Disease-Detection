##🌱 Plant Disease Detection

📝 Project Overview

This project implements a deep learning model for classifying plant diseases using the PlantVillage dataset. By leveraging a Convolutional Neural Network (CNN) built with TensorFlow/Keras, the model identifies various plant diseases from leaf images, enabling early detection and supporting agricultural management.



✨ Features





Dataset: Utilizes the PlantVillage dataset with 38 classes of plant disease and healthy leaf images in color, grayscale, and segmented formats.



Model: A robust CNN trained for high-accuracy plant disease classification.



Data Preprocessing: Includes image resizing, normalization, and data augmentation for improved model performance.



Evaluation: Visualizes training and validation accuracy/loss to track model performance.



Prediction: Enables classification of new leaf images using a custom prediction function.



Reproducibility: Sets random seeds for consistent results across runs.



🛠️ Prerequisites

To run this project, ensure you have the following installed:





Python 3.x



TensorFlow



NumPy



Matplotlib



Pillow



Kaggle API (for dataset download)



Jupyter Notebook (to run the .ipynb file)

Additionally, a Kaggle account and API credentials (kaggle.json) are required to download the dataset.



🚀 Installation





Clone the Repository:

git clone <repository-url>
cd <repository-directory>



Install Dependencies:

pip install tensorflow numpy matplotlib pillow kaggle



Set Up Kaggle API:





Obtain your Kaggle API credentials (kaggle.json) from your Kaggle account.



Place kaggle.json in the project directory or configure environment variables:

export KAGGLE_USERNAME=<your-username>
export KAGGLE_KEY=<your-api-key>



📊 Dataset

The project uses the PlantVillage dataset, which includes:





Color: RGB images of plant leaves.



Grayscale: Grayscale versions of the images.



Segmented: Images with backgrounds removed.

The dataset is automatically downloaded and extracted within the notebook using the Kaggle API.



📖 Usage





Run the Jupyter Notebook: Launch DL_aat.ipynb in Jupyter Notebook and execute cells sequentially:

jupyter notebook DL_aat.ipynb



Key Notebook Steps:





Seed Initialization: Ensures reproducibility using random, NumPy, and TensorFlow seeds.



Dataset Download: Downloads and extracts the PlantVillage dataset via the Kaggle API.



Data Preprocessing: Loads, preprocesses, and augments images for training/validation.



Model Training: Trains the CNN and visualizes accuracy/loss metrics.



Prediction: Provides functions to classify new leaf images.



Model Saving: Saves the trained model as plant_disease_model.h5 and class indices as class_indices.json.



Predicting on New Images: Use the predict_image_class function to classify new leaf images:

predicted_class = predict_image_class(model, 'path/to/image.jpg', class_indices)
print(f'Predicted class: {predicted_class}')



🧠 Model Architecture

The CNN model, built with TensorFlow/Keras, includes:





Convolutional Layers: Extract key features from leaf images.



Pooling Layers: Reduce spatial dimensions for efficiency.



Dense Layers: Perform classification across 38 disease classes.



Softmax Activation: Outputs probabilities for multi-class classification.



📈 Results





The notebook generates plots for training and validation accuracy/loss to evaluate model performance.



The trained model is saved as plant_disease_model.h5, with class indices stored in class_indices.json for easy prediction.

🤝 Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bug fixes.



🙏 Acknowledgments





The PlantVillage dataset is provided by Abdallah Ali under the CC-BY-NC-SA-4.0 license.



Built with TensorFlow, Keras, and other open-source libraries.
