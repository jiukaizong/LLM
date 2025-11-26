export default function DataTable({ columns = [], rows = [], total = 0, page = 1, pages = 1, onPage }) {
  return (
    <div className="space-y-2">
      <div className="text-sm text-gray-600">Total: {total} · Page {page}/{pages}</div>

      <div className="overflow-auto border rounded-xl">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-100 sticky top-0">
            <tr>
              {columns.map(col => (
                <th key={col} className="text-left px-3 py-2 font-semibold border-b">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="odd:bg-white even:bg-gray-50">
                {columns.map(c => (
                  <td key={c} className="px-3 py-2 border-b">{String(r[c] ?? "")}</td>
                ))}
              </tr>
            ))}
            {rows.length === 0 && (
              <tr><td className="px-3 py-6 text-center text-gray-500" colSpan={columns.length || 1}>No data</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center gap-2">
        <button className="px-3 py-1 border rounded" disabled={page <= 1} onClick={() => onPage(page - 1)}>Prev</button>
        <button className="px-3 py-1 border rounded" disabled={page >= pages} onClick={() => onPage(page + 1)}>Next</button>
      </div>
    </div>
  );
}
