"""
Central event log server for the Mycelium fleet.
Run: python event_log_server.py
Env vars: MYCELIUM_LOG_SECRET, MYCELIUM_LOG_FILE (default: events.jsonl)
"""
import json
import os
import sys
from datetime import datetime, timezone
from flask import Flask, request, jsonify

app = Flask(__name__)
_API_KEY = os.environ.get("MYCELIUM_LOG_SECRET", "")
_LOG_FILE = os.getenv("MYCELIUM_LOG_FILE", "events.jsonl")


@app.post("/event")
def receive_event():
    if not _API_KEY or request.headers.get("X-Api-Key") != _API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    event = request.get_json(silent=True)
    if not event:
        return jsonify({"error": "bad request"}), 400
    event["server_received_at"] = datetime.now(timezone.utc).isoformat()
    with open(_LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    if not _API_KEY:
        print("ERROR: MYCELIUM_LOG_SECRET not set", file=sys.stderr)
        sys.exit(1)
    app.run(host="0.0.0.0", port=int(os.getenv("MYCELIUM_LOG_PORT", "5000")))