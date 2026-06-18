from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import traceback

app = Flask(__name__)
CORS(app)

print("Загрузка EasyOCR...")
reader = easyocr.Reader(['ru'], gpu=False, verbose=False)
print("EasyOCR успешно загружен")


def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))

    # Оттенки серого
    img = img.convert("L")

    # Увеличение изображения
    width, height = img.size
    img = img.resize((width * 2, height * 2), Image.LANCZOS)

    # Контраст
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Резкость
    sharpener = ImageEnhance.Sharpness(img)
    img = sharpener.enhance(2.0)

    # Небольшое сглаживание
    img = img.filter(ImageFilter.SMOOTH_MORE)

    return np.array(img)


@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Файл не найден'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400

        img_bytes = file.read()

        processed = preprocess_image(img_bytes)

        results = reader.readtext(
            processed,
            detail=1,
            paragraph=False
        )

        if not results:
            return jsonify({
                'text': '',
                'confidence': 0,
                'details': [],
                'message': 'Текст не найден'
            })

        # Сортировка строк сверху вниз
        results.sort(key=lambda x: x[0][0][1])

        filtered = []

        for item in results:
            text = item[1].strip()

            if len(text) >= 2:
                filtered.append(item)

        results = filtered

        if not results:
            return jsonify({
                'text': '',
                'confidence': 0,
                'details': [],
                'message': 'Текст не найден'
            })

        full_text = '\n'.join(item[1] for item in results)

        details = []
        total_conf = 0

        for item in results:
            conf = round(item[2] * 100, 1)

            details.append({
                'text': item[1],
                'confidence': conf
            })

            total_conf += item[2]

        avg_conf = round((total_conf / len(results)) * 100, 1)

        return jsonify({
            'text': full_text,
            'confidence': avg_conf,
            'details': details
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            'error': str(e),
            'message': 'Ошибка сервера'
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
