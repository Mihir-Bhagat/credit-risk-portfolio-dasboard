"""A tiny local OpenAI-compatible chat-completions server used to test the LLM adapter at the HTTP level.

It records every request (headers + body) so tests can assert exactly what would leave the machine, and it can
simulate rate limits, server errors, timeouts, malformed output and a model that tries to misbehave.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class FakeLLM:
    def __init__(self):
        self.requests: list[dict] = []
        self.behaviour = "ok"          # ok | 429 | 500 | 401 | timeout | malformed | not_json | bad_metric | decline | 400_json_mode | injected_field
        self.plans: dict[str, dict] = {}  # question -> plan to return when behaviour == ok
        self.default_plan: dict | None = None
        self.fail_first_n = 0
        self._server: HTTPServer | None = None

    # ---- lifecycle
    def start(self) -> str:
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                beh = outer.behaviour
                if outer.fail_first_n > 0:
                    outer.fail_first_n -= 1
                    beh = "500"
                if beh == "429":
                    return self._send(429, {"error": {"message": "rate limit"}})
                if beh == "500":
                    return self._send(500, {"error": {"message": "boom"}})
                if beh == "401":
                    return self._send(401, {"error": {"message": "bad key"}})
                if beh == "timeout":
                    time.sleep(3)
                    return self._send(200, {})
                if beh == "400_json_mode" and "response_format" in body:
                    return self._send(400, {"error": {"message": "response_format is not supported by this model"}})
                user = json.loads(body["messages"][1]["content"])
                q = user["question"]
                if beh == "malformed":
                    return self._send(200, {"unexpected": True})
                if beh == "not_json":
                    return self._completion("Sure! Here is your answer: 42.")
                if beh == "bad_metric":
                    return self._completion(json.dumps({"answerable": True, "metric": "drop_table_users", "group_by": [], "filters": []}))
                if beh == "injected_field":
                    return self._completion(json.dumps({"answerable": True, "metric": "default_rate", "group_by": ["customer_id"], "filters": []}))
                if beh == "decline":
                    return self._completion(json.dumps({"answerable": False, "reason": "This asks about the future."}))
                plan = outer.plans.get(q, outer.default_plan)
                if plan is None:
                    plan = {"answerable": False, "reason": "no canned plan"}
                if beh == "fenced":
                    return self._completion("```json\n" + json.dumps(plan) + "\n```")
                return self._completion(json.dumps(plan))

            def _completion(self, content):
                self._send(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})

            def _send(self, code, obj):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
