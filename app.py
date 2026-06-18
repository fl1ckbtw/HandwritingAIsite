from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import easyocr
import cv2
import numpy as np
import io
import re
import traceback

app = Flask(__name__)
CORS(app)

LANGS = ['ru', 'en']
MIN_CONFIDENCE = 0.2

OCR_KWARGS = dict(
    detail=1,
    paragraph=False,
    contrast_ths=0.1,
    adjust_contrast=0.6,
    text_threshold=0.55,
    low_text=0.35,
    link_threshold=0.35,
    mag_ratio=1.8,
    decoder='beamsearch',
    beamWidth=8,
)

GARBAGE_RE = re.compile(r'^[\{\}\[\]\(\)\|\\/_\-=+*#@$%^&`~<>\'\"«».,:;!?]+$', re.UNICODE)

print('Загрузка EasyOCR...')
reader = easyocr.Reader(LANGS, gpu=False, verbose=False)
print('EasyOCR готов')


def load_image(img_bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Не удалось прочитать изображение')
    return img


def upscale(img, min_side=1600):
    h, w = img.shape[:2]
    scale = max(1.0, min_side / max(h, w))
    if scale <= 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def preprocess_variants(img_bytes):
    """Несколько вариантов предобработки — выбираем лучший по качеству OCR."""
    bgr = upscale(load_image(img_bytes))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=8, templateWindowSize=7, searchWindowSize=21)

    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8,
    )

    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(denoised, -1, kernel)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    return [
        ('color', bgr),
        ('enhanced', denoised),
        ('sharp', sharp),
        ('binary', binary),
    ]


def is_valid_fragment(text, confidence):
    text = text.strip()
    if not text or confidence < MIN_CONFIDENCE:
        return False
    if GARBAGE_RE.match(text):
        return False

    letters = re.findall(r'[\wа-яА-ЯёЁ]', text, re.UNICODE)
    if not letters:
        return False

    letter_ratio = len(letters) / len(text)
    if letter_ratio < 0.55:
        return False

    if len(text) <= 3 and (len(letters) < 2 or confidence < 0.65):
        return False

    return True


def bbox_center(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return sum(xs) / 4, sum(ys) / 4


def bbox_height(bbox):
    ys = [p[1] for p in bbox]
    return max(ys) - min(ys)


def order_results(results):
    valid = [r for r in results if is_valid_fragment(r[1], r[2])]
    if not valid:
        return []

    valid.sort(key=lambda r: bbox_center(r[0])[1])

    lines = [[valid[0]]]
    for item in valid[1:]:
        _, cy = bbox_center(item[0])
        prev_line = lines[-1]
        _, prev_cy = bbox_center(prev_line[-1][0])
        avg_h = sum(bbox_height(b[0]) for b in prev_line) / len(prev_line)
        threshold = max(avg_h * 0.55, 12)

        if abs(cy - prev_cy) <= threshold:
            prev_line.append(item)
        else:
            lines.append([item])

    ordered = []
    for line in lines:
        line.sort(key=lambda r: bbox_center(r[0])[0])
        ordered.extend(line)
    return ordered


def score_results(results):
    ordered = order_results(results)
    if not ordered:
        return 0.0
    conf_sum = sum(r[2] for r in ordered)
    text_len = sum(len(r[1].strip()) for r in ordered)
    return conf_sum * (1 + text_len * 0.05)


def recognize_best(img_bytes):
    best_results = []
    best_score = 0.0
    best_variant = 'color'

    for name, image in preprocess_variants(img_bytes):
        try:
            results = reader.readtext(image, **OCR_KWARGS)
        except Exception:
            continue

        score = score_results(results)
        if score > best_score:
            best_score = score
            best_results = results
            best_variant = name

    return order_results(best_results), best_variant


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Файл не найден'}), 400

        file = request.files['image']
        if not file.filename:
            return jsonify({'error': 'Файл не выбран'}), 400

        img_bytes = file.read()
        if not img_bytes:
            return jsonify({'error': 'Пустой файл'}), 400

        results, variant = recognize_best(img_bytes)

        if not results:
            return jsonify({
                'text': '',
                'confidence': 0,
                'details': [],
                'message': 'Текст не найден. Попробуйте более чёткое фото с хорошим освещением.',
            })

        full_text = '\n'.join(item[1].strip() for item in results)
        details = []
        total_conf = 0.0

        for item in results:
            conf = round(item[2] * 100, 1)
            details.append({'text': item[1].strip(), 'confidence': conf})
            total_conf += item[2]

        avg_conf = round((total_conf / len(results)) * 100, 1)

        return jsonify({
            'text': full_text,
            'confidence': avg_conf,
            'details': details,
            'variant': variant,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'message': 'Ошибка сервера'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
