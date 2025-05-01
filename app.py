from flask import Flask, render_template, jsonify, request
import json, pickle, torch
from speech_translate import (
    speech_to_speech_translation,
    translate_sentence,
    Vocabulary,
    Transformer,
    PAD_TOKEN
)

app = Flask(__name__, static_url_path="/static")

# Load model and vocab once
def load_model_separately():
    with open("model/transformer-model-best_src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("model/transformer-model-best_tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)
    with open("model/transformer-model-best_hyperparams.json", "r") as f:
        params = json.load(f)

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=params["d_model"],
        n_layers=params["n_layers"],
        n_heads=params["n_heads"],
        dim_feedforward=params["dim_feedforward"],
        dropout=params["dropout"],
        src_pad_idx=src_vocab[PAD_TOKEN],
        tgt_pad_idx=tgt_vocab[PAD_TOKEN],
    )
    model.load_state_dict(torch.load("model/transformer-model-best_weights.pt", map_location="cpu"))
    model.eval()
    return model, src_vocab, tgt_vocab

# Load once globally
model, src_vocab, tgt_vocab = load_model_separately()
DEVICE = torch.device("cpu")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run-translation")
@app.route("/run-translation")
def run_translation():
    print(">>> /run-translation called")
    try:
        german_text, translated_text, _ = speech_to_speech_translation(return_text=True)
        print("German:", german_text)
        print("English:", translated_text)
        return jsonify({
            "german": german_text,
            "translation": translated_text,
            "audio_url": "/static/translated_audio.mp3"
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({
            "german": None,
            "translation": None,
            "error": str(e)
        }), 500

@app.route("/translate-text")
def translate_text():
    sentence = request.args.get("sentence", "")
    if not sentence.strip():
        return jsonify({"translation": "[No sentence provided]"}), 400

    try:
        translation_tokens, _ = translate_sentence(model, sentence, src_vocab, tgt_vocab, DEVICE)
        if "<eos>" in translation_tokens:
            translation_tokens = translation_tokens[:translation_tokens.index("<eos>")]
        return jsonify({"translation": " ".join(translation_tokens)})
    except Exception as e:
        print("Text translation error:", e)
        return jsonify({"translation": None, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
