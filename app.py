from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import os

app = Flask(__name__)

# ----------------------
# Load model and data
# ----------------------
model = load_model("model.keras")  # تأكدي من اسم الملف صحيح
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
max_length = int(pickle.load(open("max_length.pkl", "rb")))

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
        # encode and pad input
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        # predict next word
        yhat = model.predict([image, sequence], verbose=0)[0]
        yhat_index = np.argmax(yhat)
        word = idx_to_word(yhat_index, tokenizer)

        # stop if word not found or end tag
        if word is None or word == "<end>":
            break

        # prevent repeating the last word
        if word == in_text.split()[-1]:
            break

        # append word
        in_text += " " + word

    # remove start and end tags
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
        file = request.files["image"]
        if file:
            # ensure uploads folder exists
            os.makedirs("static/uploads", exist_ok=True)
            image_path = os.path.join("static/uploads", file.filename)
            file.save(image_path)

            # extract features and generate caption
            photo = extract_features(image_path)
            caption = predict_caption(model, photo, tokenizer, max_length)
            image_file = file.filename

    return render_template("index.html", caption=caption, image_file=image_file)

# ----------------------
# Run app
# ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
