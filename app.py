import os
import requests
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

app = Flask(__name__)

# ---------------------------------------------------------
# وظيفة التحميل التلقائي للملفات الكبيرة
# ---------------------------------------------------------
def download_file_if_missing(url, filename):
    if not os.path.exists(filename):
        print(f"File {filename} missing. Downloading from Google Drive...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # التأكد من أن الرابط يعمل
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Download {filename} complete!")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

# استبدلي الـ IDs بالتي حصلتِ عليها من Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=اكتبي_هنا_ID_ملف_الـ_keras"
FEATURES_URL = "https://drive.google.com/uc?export=download&id=اكتبي_هنا_ID_ملف_الـ_pkl"

# تحميل الملفات قبل بدء السيرفر
download_file_if_missing(MODEL_URL, "model.keras")
# ملحوظة: الكود الخاص بكِ لا يستخدم features.pkl في الـ app.py 
# ولكنه يستخدمه في التدريب، إذا كنتِ ستحتاجينه أضيفي السطر التالي:
# download_file_if_missing(FEATURES_URL, "features .pkl")

# ----------------------
# Load model and data
# ----------------------
# قمنا بوضع التحميل داخل try/except لتجنب انهيار السيرفر إذا فشل التحميل
try:
    model = load_model("model.keras")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    max_length = int(pickle.load(open("max_length.pkl", "rb")))
except Exception as e:
    print(f"Error loading models: {e}")

# Load VGG16 feature extractor
base_model = VGG16()
vgg_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

# ----------------------
# Helper functions
# ----------------------
def extract_features(filename):
    """Extract features from an image using VGG16"""
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    """Convert integer index to word"""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    """Generate caption for the given image"""
    in_text = "<start>"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([image, sequence], verbose=0)[0]
        yhat_index = np.argmax(yhat)
        word = idx_to_word(yhat_index, tokenizer)

        if word is None or word == "<end>":
            break
        
        if word == in_text.split()[-1]:
            break

        in_text += " " + word

    final_caption = in_text.replace("<start>", "").replace("<end>", "").strip()
    return final_caption

# ----------------------
# Flask routes
# ----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_file = None

    if request.method == "POST":
        if 'image' not in request.files:
            return render_template("index.html", caption="No image uploaded")
        
        file = request.files["image"]
        if file.filename == '':
            return render_template("index.html", caption="No image selected")

        if file:
            os.makedirs("static/uploads", exist_ok=True)
            image_path = os.path.join("static/uploads", file.filename)
            file.save(image_path)

            photo = extract_features(image_path)
            caption = predict_caption(model, photo, tokenizer, max_length)
            image_file = file.filename

    return render_template("index.html", caption=caption, image_file=image_file)

# ----------------------
# Run app
# ----------------------
if __name__ == "__main__":
    # نستخدم port من المتغيرات البيئية ليتوافق مع Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)