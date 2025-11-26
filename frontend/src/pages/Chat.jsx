// src/pages/Chat.jsx
import React, { useEffect, useRef, useState } from "react";

export default function Chat() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState([]); 
  const [loading, setLoading] = useState(false);
  const [useData, setUseData] = useState(true);          
  const [extraCtx, setExtraCtx] = useState("");          
  const listRef = useRef(null);

 
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [history, loading]);

  async function send() {
    const msg = input.trim();
    if (!msg || loading) return;

    setLoading(true);
    setHistory(h => [...h, { role: "user", content: msg }]);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: msg,
          use_data: useData,
          extra_context: extraCtx || undefined,
        }),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} ${text}`);
      }

      const data = await res.json();
      const reply = (data && data.reply) ? data.reply : "(no reply)";
      setHistory(h => [...h, { role: "assistant", content: reply }]);
    } catch (err) {
      setHistory(h => [...h, { role: "assistant", content: "Request failed. " + String(err) }]);
    } finally {
      setInput("");
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="flex flex-col gap-4 max-w-6xl mx-auto">
      <h2 className="text-xl font-semibold">Risk & Performance Assistant</h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 bg-white rounded-2xl shadow p-4 flex flex-col min-h-[520px]">
          <div
            ref={listRef}
            className="flex-1 overflow-y-auto space-y-3 pr-1"
            style={{ scrollbarGutter: "stable" }}
          >
            {history.length === 0 && (
              <div className="text-gray-500 text-sm">
                Tips: Ask data-grounded questions like
                <ul className="list-disc ml-5 mt-1">
                  <li>Which departments have highest attrition?</li>
                  <li>Which credit limit bins show highest default?</li>
                  <li>Give actions to reduce attrition and default.</li>
                </ul>
              </div>
            )}

            {history.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[80%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm ${
                    m.role === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-900"
                  }`}
                >
                  <b className="block mb-1">{m.role === "user" ? "You" : "Assistant"}</b>
                  {m.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="text-gray-500 text-sm italic">Assistant is thinking…</div>
            )}
          </div>

          <div className="mt-3 flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask about performance & risk..."
              className="flex-1 border rounded-xl p-3 text-sm focus:outline-none focus:ring"
              rows={2}
            />
            <button
              onClick={send}
              disabled={loading || !input.trim()}
              className={`px-4 py-2 rounded-xl text-sm font-medium ${
                loading || !input.trim()
                  ? "bg-gray-200 text-gray-500"
                  : "bg-blue-600 text-white hover:bg-blue-700"
              }`}
            >
              {loading ? "Thinking..." : "Send"}
            </button>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow p-4">
          <h3 className="font-medium mb-3">Context Control</h3>

          <label className="flex items-center gap-2 text-sm mb-3 select-none">
            <input
              type="checkbox"
              checked={useData}
              onChange={(e) => setUseData(e.target.checked)}
            />
            Use CSV data context (recommended)
          </label>

          <div className="mb-2 text-xs text-gray-500">
            Optional: paste current page stats/filters here (will be appended into context).
          </div>
          <textarea
            value={extraCtx}
            onChange={(e) => setExtraCtx(e.target.value)}
            placeholder="e.g., Department = Sales, OverTime = Yes, Top-3 risky segments from Employee page..."
            className="w-full border rounded-xl p-2 text-sm min-h-[140px] focus:outline-none focus:ring"
          />

          <div className="mt-3">
            <button
              onClick={() =>
                setExtraCtx(
                  "Example extra context:\n- Department filter: Sales\n- OverTime: Yes\n- Top-3 risky segments observed on dashboard..."
                )
              }
              className="text-xs text-blue-600 hover:underline"
              type="button"
            >
              Insert example
            </button>
          </div>

          <div className="mt-4 text-xs text-gray-500">
            Backend endpoint: <code>/api/chat</code><br />
            It expects: <code>{`{ message, use_data, extra_context? }`}</code>
          </div>
        </div>
      </div>
    </div>
  );
}
