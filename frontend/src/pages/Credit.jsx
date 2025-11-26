import { useEffect, useState } from "react";
import { getCreditMetrics, getCreditData } from "../lib/api";
import DataTable from "../components/DataTable";

export default function Credit() {
  const [metrics, setMetrics] = useState(null);
  const [err, setErr] = useState("");
  const [data, setData] = useState({
    columns: [],
    rows: [],
    total: 0,
    page: 1,
    pages: 1,
  });
  const [pageSize] = useState(50);

  useEffect(() => {
    getCreditMetrics()
      .then(setMetrics)
      .catch((e) => setErr(e.message || String(e)));
  }, []);

  useEffect(() => {
    loadPage(1);
  }, []);

  function loadPage(p) {
    const offset = (p - 1) * pageSize;
    getCreditData({ offset, limit: pageSize })
      .then(setData)
      .catch((e) => setErr(e.message || String(e)));
  }

  if (err) return <div className="text-red-600">Error: {err}</div>;
  if (!metrics) return <div>Loading...</div>;

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Credit Metrics</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Stat label="Rows" value={metrics.rowcount} />
        <Stat
          label="Default Rate"
          value={`${(metrics.default_rate * 100).toFixed(2)}%`}
        />
        <Stat
          label="Avg Credit Limit"
          value={Number(metrics.avg_limit).toFixed(0)}
        />
      </div>

      <div className="pt-2">
        <b>PAY_0 distribution (top 5)</b>
        <ul className="list-disc ml-5">
          {Object.entries(metrics.pay0_dist_top5 || {}).map(([k, v]) => (
            <li key={k}>
              {k}: {(v * 100).toFixed(2)}%
            </li>
          ))}
        </ul>
      </div>

      <div className="space-y-3 pt-6">
        <h3 className="text-lg font-semibold">Raw Data</h3>
        <DataTable
          columns={data.columns}
          rows={data.rows}
          total={data.total}
          page={data.page}
          pages={data.pages}
          onPage={(p) => loadPage(p)}
        />
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="p-4 rounded-xl border bg-gray-50">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}
