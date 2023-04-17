import base64
import io
import os
import requests
import redis
from pprint import pprint
from flask import abort, Request, jsonify



r_host = os.environ.get('R_HOST')
r_port = os.environ.get('R_PORT')
r_pass = os.environ.get('R_PASS')
alpr_token = os.environ.get('ALPR_TOKEN')
r = redis.StrictRedis(host=r_host, port=r_port, password=r_pass)


def plate_recognizer(plate: str):
    bytes_data = base64.b64decode(plate)
    buffer = io.BufferedReader(io.BytesIO(bytes_data))
    regions = ['id']

    response = requests.post(
        'https://api.platerecognizer.com/v1/plate-reader/',
        data=dict(regions=regions),
        files=dict(upload=buffer),
        headers={'Authorization': f'Token {alpr_token}'})
    response_json = response.json()

    pprint(response_json)
    plate_number = response_json['results'][0]['plate']
    return plate_number


def validate_plate(request: Request):
    api_key = os.environ.get('API_KEY')

    if not api_key:
        abort(500, 'Internal server error')

    if request.headers.get('X-API-KEY') != api_key:
        abort(401, 'Unauthorized - Incorrect token')

    request_json = request.get_json()

    if not request_json and not 'data' in request_json:
        abort(400, 'Bad Request, missing data')

    plate_number = plate_recognizer(request_json['data'])

    status = r.get(plate_number)

    response = {'plate': plate_number, 'status': "valid" if status else "invalid"}

    return jsonify(response)

