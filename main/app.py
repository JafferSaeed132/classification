from flask import Flask, render_template, request
import numpy as np
import base64
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('my_model.pt')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(image_stream):
    # Read and decode the image
    img_bytes = image_stream.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Perform detection
    results = model.predict(image, conf=0.5)
    
    # Process results
    for r in results:
        annotated_image = r.plot(conf=True)  # Image with bounding boxes
    
    return annotated_image

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')
            
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
            
        if file and allowed_file(file.filename):
            # Get original image as base64
            file.seek(0)
            original_img_base64 = base64.b64encode(file.read()).decode('utf-8')
            
            # Process image for detection
            file.seek(0)
            detection_img = predict_on_image(file)
            
            # Convert detection image to base64
            _, buffer = cv2.imencode('.png', detection_img)
            detection_img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return render_template('result.html',
                                original_img_data=original_img_base64,
                                detection_img_data=detection_img_base64)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)