import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
model = pickle.load(open("Image_classifier.pkl", "rb"))

@app.route('/')
def home():
    return render_template('Images.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        img_gen = request.files["files"]
        img = Image.open(img_gen)
        
        with BytesIO() as buf:
            img.save(buf, 'jpeg')
            image_bytes = buf.getvalue()
        encoded_string = base64.b64encode(image_bytes).decode() 

        img_gen = request.files["files"]
        img = Image.open(img_gen)
        img = img.resize((32, 32))
        img = np.ravel(img)
        img_arr = np.array([img])
        prediction = model.predict(img_arr)

        output = ["a dog!", "a cat!", "a panda!"][prediction[0]]

        return render_template('Images.html', img_data = encoded_string, prediction_text=f'Image is {output}')
    
    except:
        return render_template('Images.html')

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)

    img = Image.open(data.values()[0])
    img = img.resize((32, 32))
    img = np.ravel(img)
    img_arr = np.array([img])
    prediction = model.predict(img_arr)

    output = ["a dog!", "a cat!", "a panda!"][prediction[0]]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)