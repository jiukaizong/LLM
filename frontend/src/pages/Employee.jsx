import React, { useEffect, useState } from "react";

export default function Employee() {
  const [meta, setMeta] = useState(null);     
  const [rows, setRows] = useState([]);       
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [offset, setOffset] = useState(0);
  const limit = 50;

  async function loadPage(nextOffset = 0) {
    setLoading(true);
    setErr("");
    try {
      const res = await fetch(`/api/employee/data?offset=${nextOffset}&limit=${limit}`);
      const data = await res.json();

      console.log("employee/data response:", data);

      if (!data || !Array.isArray(data.rows)) {
        throw new Error("Unexpected API shape: missing rows");
      }
      setRows(data.rows);
      setMeta({
        columns: data.columns || (data.rows[0] ? Object.keys(data.rows[0]) : []),
        total: data.total ?? 0,
        hasNext: !!data.has_next,
        hasPrev: !!data.has_prev,
        offset: data.offset ?? nextOffset,
        limit: data.limit ?? limit,
        page: data.page ?? null,
        pageSize: data.page_size ?? null,
      });
      setOffset(nextOffset);
    } catch (e) {
      console.error(e);
      setErr(e.message || "Load failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadPage(0);
  }, []);

  const DISPLAY_COLS = 10;
  const allCols = meta?.columns || [];
  const columnsToShow = allCols.slice(0, DISPLAY_COLS);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Employee (HR)</h2>
        <div className="text-sm text-gray-500">
          {loading ? "Loading..." : `Rows: ${meta?.total ?? 0}`}
        </div>
      </div>

      {err && (
        <div className="p-3 rounded-lg bg-red-50 text-red-700 border border-red-200">
          {err}
        </div>
      )}

      <details className="text-xs text-gray-500">
        <summary>Debug</summary>
        <pre className="whitespace-pre-wrap break-words">
{JSON.stringify(
  { columns: meta?.columns, total: meta?.total, hasNext: meta?.hasNext, hasPrev: meta?.hasPrev, offset, limit },
  null,
  2
)}
        </pre>
      </details>

      <div className="overflow-auto border rounded-xl">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-100 sticky top-0">
            <tr>
              {columnsToShow.map((c) => (
                <th key={c} className="px-3 py-2 text-left font-semibold border-b">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {!loading && rows.length === 0 && (
              <tr>
                <td className="px-3 py-3 text-gray-500" colSpan={columnsToShow.length}>
                  No data.
                </td>
              </tr>
            )}
            {rows.map((r, i) => (
              <tr key={i} className="odd:bg-white even:bg-gray-50">
                {columnsToShow.map((c) => (
                  <td key={c} className="px-3 py-2 border-b">
                    {String(r[c] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* next page */}
      <div className="flex items-center gap-2">
        <button
          className="px-3 py-1 rounded-lg border"
          onClick={() => loadPage(Math.max(0, offset - limit))}
          disabled={loading || !(meta?.hasPrev)}
        >
          Prev
        </button>
        <button
          className="px-3 py-1 rounded-lg border"
          onClick={() => loadPage(offset + limit)}
          disabled={loading || !(meta?.hasNext)}
        >
          Next
        </button>
        <span className="text-xs text-gray-500">
          offset {offset} ~ {offset + rows.length - 1}
        </span>
      </div>
    </div>
  );
}
