from flask import Flask, render_template, request
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

model_path = "model_weights"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

test_dataset_path = "Dataset/test_data.csv"
test_df = pd.read_csv(test_dataset_path)

def predict_rating(review_text):
 
    inputs = tokenizer(review_text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_rating = logits.item()

    return predicted_rating

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "review_text" in request.form:
            review_text = request.form["review_text"]
            predicted_rating = predict_rating(review_text)
            return render_template("index.html", review_text=review_text, predicted_rating=predicted_rating)

        elif "index_number" in request.form:
            index_number = int(request.form["index_number"])
            review_text = test_df.loc[index_number, "reviewText"]
            actual_rating = test_df.loc[index_number, "Overall"]
            predicted_rating = predict_rating(review_text)
            return render_template("index.html", review_text=review_text, actual_rating=actual_rating, predicted_rating=predicted_rating)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
