import { useState } from "react";
import { postInsights } from "../lib/api";

export default function Insights(){
  const [qs, setQs] = useState(["What are the top risks this month?"]);
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  async function run(){
    setLoading(true);
    setErr("");
    try {
      const out = await postInsights({
        sources:["employee","credit"],
        questions: qs,
        controls:{}
      });
      setResp(out);
    } catch(e){
      setErr(e.message || "Failed to fetch insights");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">LLM Insights</h2>

      <textarea
        className="border p-2 w-full h-28 rounded-lg"
        value={qs.join("\n")}
        onChange={e=>setQs(e.target.value.split("\n"))}
        placeholder="Enter one question per line…"
      />

      <div className="flex gap-2">
        <button
          className="px-4 py-2 bg-black text-white rounded-lg disabled:opacity-60"
          onClick={run}
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Generate Insights"}
        </button>
      </div>

      {err && <div className="text-red-600">{err}</div>}

      {resp && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Panel title="Summary"><p>{resp.summary}</p></Panel>
          <Panel title="Risks">
            <ul className="list-disc ml-5">{(resp.risks||[]).map((r,i)=><li key={i}>{r}</li>)}</ul>
          </Panel>
          <Panel title="Actions">
            <ul className="list-disc ml-5">{(resp.actions||[]).map((a,i)=><li key={i}>{a}</li>)}</ul>
          </Panel>
          <Panel title="Caveats">
            <ul className="list-disc ml-5">{(resp.caveats||[]).map((c,i)=><li key={i}>{c}</li>)}</ul>
          </Panel>
        </div>
      )}
    </div>
  );
}

function Panel({title, children}){
  return (
    <div className="p-4 rounded-xl border bg-gray-50">
      <div className="font-semibold mb-2">{title}</div>
      <div className="text-sm leading-relaxed">{children}</div>
    </div>
  )
}
