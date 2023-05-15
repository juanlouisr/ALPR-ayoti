from flask import Flask, request, jsonify
from plate_recognizer import *

app = Flask(__name__)

model = load_model()

@app.route('/inference', methods=['POST'])
def inference():
    # Retrieve data from the request
    request_json = request.get_json()

    if not request_json and not 'data' in request_json:
        return jsonify({'error': 'missing data'}), 400

    # Perform inference or any other processing on the data
    result = process_data(request_json)

    # Return the result as a JSON response
    return jsonify(result)

def process_data(request_json):

    file_path, image, image_hash = process_image(request_json['data'])
    predictions = detect_objects(model, file_path)
    cropped_path, cropped_image = save_cropped_image(image, predictions[:, :4], image_hash)
    ocr_result = easy_ocr(cropped_image)
    print(ocr_result)
    processed_result = {'status': 'success'}
    return processed_result

if __name__ == '__main__':
    app.run()