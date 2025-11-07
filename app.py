# app.py
import os
import time
from collections import deque, defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # set this in Render / Railway / env
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # change if needed
MAX_HISTORY_LINES = 6  # include a short history context if you later pass conversation history

# Simple rate limit config (per IP)
RATE_LIMIT_WINDOW = 60 * 60    # seconds (1 hour)
RATE_LIMIT_MAX = 120           # max requests per window per IP (adjust as needed)

# --- App init ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # restrict origin in production if needed
client = OpenAI(api_key=OPENAI_API_KEY)

# In-memory rate limit store: { ip: deque([timestamps]) }
request_log = defaultdict(lambda: deque())

# --- Helpers ---
def ip_from_request(req):
    # X-Forwarded-For support for production behind proxies
    if "X-Forwarded-For" in req.headers:
        return req.headers["X-Forwarded-For"].split(",")[0].strip()
    return req.remote_addr or "unknown"

def check_rate_limit(ip: str):
    now = time.time()
    q = request_log[ip]
    # remove old entries
    while q and q[0] <= now - RATE_LIMIT_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT_MAX:
        return False, RATE_LIMIT_WINDOW - (now - q[0])
    q.append(now)
    return True, None

def build_system_prompt():
    # Keep this prompt short but clear; you can expand personality details here
    return (
        "Tum 'Selection Mitra' ho — ek friendly aur motivating study helper. "
        "Tum Hindi/English/Hinglish dono me baat kar sakte ho. "
        "Har jawab ko exam-oriented, clear aur short steps me do. "
        "Agar user summary maange (e.g., '5 line summary'), to concise do. "
        "Agar user puchhe ki books, plan ya tricks, to practical aur ranked suggestions do. "
        "Hamesha polite raho aur exam-focused examples do."
    )

# Optional: simple text sanitization / length checks
def sanitize_text(s: str) -> str:
    return s.strip()

# --- Routes ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL}), 200

@app.route("/chat", methods=["POST"])
def chat():
    ip = ip_from_request(request)
    allowed, retry_after = check_rate_limit(ip)
    if not allowed:
        return jsonify({"error": "rate_limited", "retry_after_seconds": int(retry_after)}), 429

    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "invalid_request", "message": "Provide JSON body with 'message' field."}), 400

    user_message = sanitize_text(data.get("message", ""))
    if not user_message:
        return jsonify({"error": "invalid_request", "message": "Empty message."}), 400

    # Prepare conversation messages
    system_prompt = build_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Optionally accept short conversation history from frontend in data["history"]
    # (not included by default for simplicity)
    try:
        # Call OpenAI Chat Completions
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            # Optional params: temperature, max_tokens etc.
            temperature=0.2,
            max_tokens=800,
        )

        # Newer SDK returns choices with message
        reply_text = ""
        try:
            # safe extraction
            reply_text = resp.choices[0].message.content
        except Exception:
            # fallback to resp.choices[0].text if older format
            reply_text = getattr(resp.choices[0], "text", "") or "Maaf — response parsing error."

        # respond to frontend
        return jsonify({"reply": reply_text}), 200

    except Exception as e:
        # Log server-side error (in real deployment, use proper logging)
        print("OpenAI error:", str(e))
        return jsonify({"error": "openai_error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # For local dev use: python app.py
    app.run(host="0.0.0.0", port=port, debug=False)