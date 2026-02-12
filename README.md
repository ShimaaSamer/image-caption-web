# AI Image Captioning System 🤖📸

This project is an automated system designed to generate descriptive English captions for images. [cite_start]It bridges the gap between **Computer Vision (CV)** and **Natural Language Processing (NLP)**[cite: 29, 30]. [cite_start]The system is built using a Deep Learning architecture to identify visual elements and translate them into syntactically correct sentences[cite: 30].

## 🏗️ System Architecture
[cite_start]The model utilizes a **Merge-Architecture** (Encoder-Decoder) consisting of two main branches[cite: 9, 33]:

* [cite_start]**The Encoder (Visual Branch):** Uses a pre-trained **VGG16** network[cite: 31, 34]. [cite_start]The final classification layer was removed to extract a **4096-dimensional feature vector** that captures high-level semantic image content[cite: 35].
* [cite_start]**The Decoder (Language Branch):** Consists of an **Embedding layer** followed by a **Long Short-Term Memory (LSTM)** layer with 256 units to maintain sequence context[cite: 31, 36].
* [cite_start]**The Merging Block:** Combines visual and textual vectors into a unified dense layer with a **Softmax** activation to predict the next word in the sequence[cite: 38].

## 📊 Dataset & Preprocessing
* [cite_start]**Dataset:** Flickr8k[cite: 10, 15].
* [cite_start]**Image Processing:** Images are resized to $224 \times 224$ and normalized for the VGG16 backbone[cite: 41].
* [cite_start]**Text Cleaning:** Captions are converted to lowercase, stripped of special characters, and wrapped in `<start>` and `<end>` tokens[cite: 42].
* [cite_start]**Efficiency:** Features were pre-extracted and saved as `features.pkl` to reduce training time by 90%[cite: 55].

## 🚀 Training Configuration
* [cite_start]**Optimizer:** Adam Optimizer[cite: 45].
* [cite_start]**Loss Function:** Categorical Cross-Entropy[cite: 46].
* [cite_start]**Regularization:** Dropout layers (rate: 0.4) to prevent overfitting[cite: 47].
* **Epochs:** 15 | [cite_start]**Batch Size:** 32[cite: 48].

## 📂 Project Structure
| File/Folder | Description |
| :--- | :--- |
| `app.py` | Flask web application backend. |
| `static/` & `templates/` | Web interface assets and HTML layouts. |
| `model.keras` / `best_model.h5` | The trained deep learning model weights. |
| `tokenizer.pkl` | Pickled tokenizer for text-to-sequence conversion. |
| `Run_Project.ipynb` | Jupyter Notebook for testing and demonstration. |

## 🧪 Experimental Results
[cite_start]The model was evaluated using the **BLEU Score** to compare predicted captions against human references[cite: 61].

### Sample Performance:
| Image Input | Actual Caption (Human) | Predicted Caption (AI) |
| :--- | :--- | :--- |
| Two dogs playing | [cite_start]"black dog and spotted dog are fighting" [cite: 67] | [cite_start]"two dogs playing with plastic toy in the snow" [cite: 67] |
| Girl painting | [cite_start]"little girl is sitting in front of large painted rainbow" [cite: 67] | [cite_start]"two children sitting on the side of rainbow" [cite: 67] |

## 🛠️ Future Improvements
* [cite_start]**Attention Mechanisms:** To allow the model to focus on specific image regions per word[cite: 76].
* [cite_start]**Beam Search:** Replacing greedy decoding for globally optimal sentences[cite: 77].
* [cite_start]**Advanced Encoders:** Upgrading to ResNet or InceptionV3 for better resolution[cite: 78].

---
[cite_start]**Developed by:** * Shimaa Samer Ahmed Ibrahim [cite: 2, 5]
* [cite_start]Shorouk Mostafa Mohamed Hassan [cite: 3, 6]
* [cite_start]**Date:** February 2026 [cite: 11, 16]

## ✍️ Authors
* [cite_start]**Shimaa Samer** [cite: 2, 5]
* [cite_start]**Shorouk Mostafa** [cite: 3, 6]