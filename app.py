from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Define a function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# Define the home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prec')
def prec():
    return render_template('prec.html')

@app.route('/predict')
def pred():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']
    
    # Save the file to a temporary location
    file_path = 'temp.jpg'
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Make a prediction
    prediction = model.predict(img_array)
    
    # Get the predicted class probability
    predicted_probability = prediction[0][0]

    # Define the class labels
    class_labels = ["It's the case of Monkey Pox", "It's not MonkeyPox"]

    # Set a threshold to classify the image
    threshold = 0.5

    # Get the predicted class label
    if predicted_probability >= threshold:
        prediction_str = class_labels[1]  # Monkey Pox
    else:
        prediction_str = class_labels[0]  # Other
    

    # Load the image and convert it to a base64-encoded string
    with open(file_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode()

    # Render the HTML template with the prediction and the image
    return render_template('result.html', prediction=prediction_str, image=img_base64)




if __name__ == '__main__':
    app.run()   
