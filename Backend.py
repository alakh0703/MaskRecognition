from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np

app = Flask(__name__)
model = load_model("ninety_eight.h5")

def predict(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    if(prediction[0][0] > 0.5):
        print("Masked")
    else:
        print("Unmasked")
    return "Masked" if prediction[0][0] > 0.5 else "Unmasked"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Create the 'temp' directory if it doesn't exist
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Assuming you have an input field with the name 'file'
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":
            # Save the uploaded file to a temporary location
            temp_path = os.path.join(temp_dir, "temp.jpg")
            uploaded_file.save(temp_path)

            # Get the prediction result
            result = predict(temp_path)

            return jsonify({'result': result})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

