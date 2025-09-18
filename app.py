# from flask import Flask, request, render_template
# import numpy as np
# import pytesseract
# import cv2
# import re
# import joblib
# from PIL import Image
# import io

# app = Flask(__name__)

# # Load the trained model
# model = joblib.load('model.joblib')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('index.html', prediction_text="⚠️ No file uploaded.")

#     file = request.files['file']
#     if file.filename == '':
#         return render_template('index.html', prediction_text="⚠️ No file selected.")

#     try:
#         # Read image using PIL and convert to OpenCV format
#         image = Image.open(file.stream).convert('RGB')
#         img_array = np.array(image)
#         gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

#         # Extract text using pytesseract
#         extracted_text = pytesseract.image_to_string(gray)

#         if not extracted_text.strip():
#             return render_template('index.html', prediction_text="❌ No readable text found in the image.")

#         # Clean text
#         cleaned_text = re.sub(r'[^\w\s]', '', extracted_text.lower())

#         # Predict sentiment
#         prediction = model.predict([cleaned_text])[0]
#         return render_template('index.html', prediction_text=f"📄 Extracted Text:\n\n{extracted_text.strip()}\n\n🔍 Predicted Sentiment: {prediction}")

#     except Exception as e:
#         return render_template('index.html', prediction_text=f"⚠️ Error: {e}")

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
import pytesseract
import re
import joblib
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = joblib.load('model.joblib')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def extract_text_and_predict(image_path):
    gray = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(gray)
    if not extracted_text.strip():
        return "No text found."

    cleaned = re.sub(r'[^\w\s]', '', extracted_text.lower())
    prediction = model.predict([cleaned])[0]
    return f"📄 Text: {extracted_text.strip()}<br>🔍 Sentiment: <b>{prediction}</b>"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            prediction = extract_text_and_predict(path)
            return render_template('index.html', result=prediction, image_path=path)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
