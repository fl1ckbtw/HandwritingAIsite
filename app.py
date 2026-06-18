import os
import gc
import re
import traceback

import cv2
import easyocr
import numpy as np

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

CORS(app)

LANGS = ['ru']
MIN_CONFIDENCE = 0.3

_reader = None


def get_reader():
    global _reader

    if _reader is None:
        print("Loading EasyOCR...")

        _reader = easyocr.Reader(
            LANGS,
            gpu=False,
            verbose=False
        )

        print("EasyOCR loaded")

    return _reader


def load_image(img_bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)

    img = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise ValueError("Не удалось открыть изображение")

    return img


def preprocess(img_bytes):
    img = load_image(img_bytes)

    h, w = img.shape[:2]

    max_size = 1000

    if max(h, w) > max_size:

        scale = max_size / max(h, w)

        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    gray = cv2.fastNlMeansDenoising(
        gray,
        None,
        5
    )

    return gray


GARBAGE = re.compile(
    r'^[^а-яА-ЯёЁ0-9]+$'
)


def valid(text, conf):

    text = text.strip()

    if not text:
        return False

    if conf < MIN_CONFIDENCE:
        return False

    if GARBAGE.match(text):
        return False

    return True


@app.route('/')
def home():

    return send_file(
        os.path.join(
            app.root_path,
            'index.html'
        )
    )


@app.route('/health')
def health():

    return jsonify({

        'status': 'ok',

        'service': 'handwriting-ocr'

    })


@app.route('/recognize', methods=['POST'])
def recognize():

    try:

        if 'image' not in request.files:

            return jsonify({

                'error': 'Нет файла'

            }), 400


        file = request.files['image']

        img_bytes = file.read()


        if not img_bytes:

            return jsonify({

                'error': 'Пустой файл'

            }), 400


        image = preprocess(img_bytes)


        reader = get_reader()


        results = reader.readtext(

            image,

            detail=1,

            paragraph=False,

            decoder='greedy',

            batch_size=1

        )


        texts = []

        details = []

        total = 0


        for item in results:

            text = item[1].strip()

            conf = item[2]


            if valid(text, conf):

                texts.append(text)

                details.append({

                    'text': text,

                    'confidence': round(
                        conf * 100,
                        1
                    )

                })

                total += conf


        if not texts:

            del image
            del results

            gc.collect()


            return jsonify({

                'text': '',

                'confidence': 0,

                'details': [],

                'message': 'Текст не найден'

            })


        avg = round(

            total / len(texts) * 100,

            1

        )


        response = jsonify({

            'text': '\n'.join(texts),

            'confidence': avg,

            'details': details

        })


        del image
        del results

        gc.collect()


        return response


    except Exception as e:

        traceback.print_exc()

        gc.collect()

        return jsonify({

            'error': str(e)

        }), 500


if __name__ == '__main__':

    port = int(

        os.environ.get(

            'PORT',

            5000

        )

    )

    app.run(

        host='0.0.0.0',

        port=port,

        debug=False

    )
