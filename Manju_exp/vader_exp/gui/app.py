"""
Simple VADER sentiment GUI.

- A text box to enter a sample.
- "Clear" button empties the box.
- "Analyze" button shows positive / neutral / negative with their numbers.

Run:
    uv run vader_exp/gui/app.py
Then open http://localhost:8080
"""

from flask import Flask, request, jsonify, render_template_string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

PAGE = """
<!doctype html>
<title>VADER Sentiment</title>
<style>
  body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 16px; }
  textarea { width: 100%; height: 120px; font-size: 15px; padding: 10px; box-sizing: border-box; }
  button { font-size: 15px; padding: 8px 18px; margin: 12px 8px 0 0; cursor: pointer; }
  .row { display: flex; justify-content: space-between; padding: 10px 14px; margin-top: 8px;
         border-radius: 6px; font-size: 16px; }
  .positive { background: #e6f5e6; }
  .neutral  { background: #eeeeee; }
  .negative { background: #fce6e6; }
  .num { font-weight: bold; }
  #result { margin-top: 20px; }
</style>

<h2>VADER Sentiment Analyzer</h2>
<textarea id="text" placeholder="Enter sample text here..."></textarea>
<div>
  <button onclick="analyze()">Analyze</button>
  <button onclick="clearText()">Clear</button>
</div>

<div id="result"></div>

<script>
async function analyze() {
  const text = document.getElementById('text').value;
  const res = await fetch('/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });
  const d = await res.json();
  document.getElementById('result').innerHTML =
    `<div class="row positive"><span>Positive</span><span class="num">${d.pos}</span></div>
     <div class="row neutral"><span>Neutral</span><span class="num">${d.neu}</span></div>
     <div class="row negative"><span>Negative</span><span class="num">${d.neg}</span></div>
     <p>Overall: <b>${d.overall}</b> (compound ${d.compound})</p>`;
}
function clearText() {
  document.getElementById('text').value = '';
  document.getElementById('result').innerHTML = '';
}
</script>
"""


def overall_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.get_json().get("text", "")
    s = analyzer.polarity_scores(text)
    return jsonify(
        pos=s["pos"],
        neu=s["neu"],
        neg=s["neg"],
        compound=s["compound"],
        overall=overall_label(s["compound"]),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
