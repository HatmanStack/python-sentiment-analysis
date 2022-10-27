import os
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'] )
def logistic_regression():
    print('START')
    if request.method == 'GET':
      return "Hello"
    if request.method == 'POST':
      body = dict(request.get_json())
      for index,value in enumerate(body.values()):
        if isinstance(value, list):
          text = value
        else:
          hash = value
      print("Text:  {}".format(text))
    return create_logistic_regression(text, hash)

def create_logistic_regression(text, hash):
    finbert = BertForSequenceClassification.from_pretrained('./pytorch',num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('./pytorch')
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    print('S:  {}'.format(text))
    results = nlp(text)
    print("RESULTS:  {}".format(results))
    return parse_results(results, hash)

def parse_results(results, hash):
  new_dict = {'positive': 0, 'neutral': 0, 'negative': 0}
  count = [0,0,0]
  for i in results:
      key = i.get('label')
      value = i.get('score')
      if key == 'Positive':
          count[0] += 1
          new_dict['positive'] += value
      if key == 'Neutral':
          count[1] += 1
          new_dict['neutral'] += value
      if key == 'Negative':
          count[2] += 1
          new_dict['negative'] += value
  new_dict['positive'] = [count[0], (new_dict['positive'] / count[0])] if count[0] > 0 else [0,0]
  new_dict['neutral'] = [count[1], (new_dict['neutral'] / count[1])] if count[1] > 0 else [0, 0]
  new_dict['negative'] = [count[2], (new_dict['negative'] / count[2])] if count[2] > 0 else [0, 0]
  new_dict['hash'] = hash
  print(new_dict)
  return new_dict
  

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
  
  
