from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def logistic_regression():
    if request.method == 'GET':
        return "Hello"
    if request.method == 'POST':
        text = list(request.get_json().values())[0]
        hash = list(request.get_json().values())[1]
        nlp = pipeline("sentiment-analysis", model=BertForSequenceClassification.from_pretrained('./pytorch', num_labels=3), tokenizer=BertTokenizer.from_pretrained('./pytorch'))
        results = nlp(text)
        return parse_results(results, hash)

def parse_results(results, hash):
    categories = ['positive', 'neutral', 'negative']
    new_dict = {category: [0, 0] for category in categories}
    count = [0, 0, 0]
    for i in results:
        key = i.get('label')
        value = i.get('score')
        if key in categories:
            index = categories.index(key)
            count[index] += 1
            new_dict[key][0] += value
    for i in range(len(categories)):
        if count[i] > 0:
            new_dict[categories[i]][1] = new_dict[categories[i]][0] / count[i]
    new_dict['hash'] = hash
    return new_dict

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)