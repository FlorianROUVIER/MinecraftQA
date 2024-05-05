from flask import Flask, render_template, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)

# Charger le modèle et le tokenizer
model_path = "./modelsave/my_model2"

model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Obtenir la question depuis le formulaire
        data = request.get_json()
        question = data["question"]
        # Convertir la question en tokens
        inputs = tokenizer(question, return_tensors="pt")
        # Effectuer la prédiction
        outputs = model.generate(inputs.input_ids)
        # Décoder la réponse préditet'as
        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Renvoyer la réponse au format JSON
        return jsonify({"answer": predicted_answer})
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
