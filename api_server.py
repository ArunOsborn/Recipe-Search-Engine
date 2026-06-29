from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import os
import main as engine
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Load index data at startup
engine.loadData()

def get_title_from_file(file_key: str) -> str:
    try:
        if os.path.exists(file_key):
            with open(file_key, "r", encoding="utf-8") as fh:
                soup = BeautifulSoup(fh.read(), "lxml")
                if soup.title and soup.title.string:
                    return soup.title.string.strip()
                h1 = soup.find("h1")
                if h1:
                    return h1.get_text(strip=True)
    except Exception:
        pass
    return os.path.basename(file_key)


@app.route("/search")
def search():
    q = request.args.get("q") or request.args.get("query") or ""
    if not q:
        return jsonify({"error": "missing query parameter 'q'"}), 400

    results = engine.queryItems(q)
    inv = {str(v): k for k, v in engine.docID.items()}

    hits = []
    for doc_id, info in results.items():
        dstr = str(doc_id)
        score = info.get("score", 0)
        file_key = inv.get(dstr)
        meta = engine.docInfo.get(dstr, {}) if hasattr(engine, "docInfo") else {}
        name = meta.get("name") or (file_key and get_title_from_file(file_key)) or file_key or dstr
        total = meta.get("total_time")
        if total is None:
            pt = meta.get("preparation_time") or 0
            ct = meta.get("cooking_time") or 0
            if pt or ct:
                try:
                    total = int(pt) + int(ct)
                except Exception:
                    total = None

        hits.append({"name": name, "total_time": total, "score": score, "doc": file_key})

    hits.sort(key=lambda x: x.get("score", 0), reverse=True)
    return jsonify(hits[:10])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
