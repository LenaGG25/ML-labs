from flask import Flask, jsonify, render_template, request, abort
from auth import require_auth
from model import predict
from schemas import (
    PredictRequest,
    PredictResponse,
    PredictResponseItem,
)

app = Flask(__name__)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
@require_auth
def index():
    return render_template("index.html")


@app.post("/classify")
@require_auth
def classify_form():
    text = (request.form.get("text") or "").strip()
    if not text:
        abort(400, description="Пустое поле текста")
    # HTML-форма: можно обойтись без Pydantic
    result = predict([text])[0]
    return render_template("result.html", result=result)


@app.post("/predict")
@require_auth
def predict_api():
    """
    JSON:
      короткий: {"text": "..."}
      пакетный: {"items": [{"text": "..."}, {"text": "..."}]}
    Ответ:
      {"items": [{"text": "...", "label": "положительно", "prob": 0.93}]}
    """
    payload = request.get_json(silent=True) or {}

    # Нормализация короткого формата в общий
    if "text" in payload and "items" not in payload:
        payload = {"items": [{"text": str(payload["text"]).strip()}]}

    # Десериализация и валидация входа
    try:
        req = PredictRequest(**payload)
    except Exception as exc:
        abort(400, description=f"Invalid payload: {exc}")

    texts = [i.text for i in req.items if i.text.strip()]
    if not texts:
        abort(400, description="Empty items")

    # Инференс (модель возвращает dict(text,label,prob))
    preds = predict(texts)

    # Сериализация ответа через Pydantic
    resp = PredictResponse(
        items=[PredictResponseItem(**p) for p in preds]
    )
    return jsonify(resp.model_dump())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
