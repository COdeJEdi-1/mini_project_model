import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import json

# i am chahat
# Load the saved model
loaded_model = load_model("my_model.keras")

# Load class indices from the saved file
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Load the single prediction image
test_image_path = 'dataset/single_prediction/CornCommonRust3.JPG'
test_image = image.load_img(test_image_path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make prediction using the loaded model
result = loaded_model.predict(test_image)
predicted_class_index = np.argmax(result)

# Get predicted class label using the loaded class indices
predicted_class_label = list(class_indices.keys())[predicted_class_index]

print("Predicted disease:", predicted_class_label)
