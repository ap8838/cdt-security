import React, { useEffect, useState } from "react";
import { fetchMetrics } from "../api";

export default function MetricsTab() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    setLoading(true);
    fetchMetrics()
      .then((m) => setMetrics(m))
      .catch((e) => setMetrics({ error: String(e) }))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="bg-white p-4 rounded shadow">Loading metrics...</div>;

  return (
    <div className="bg-white p-4 rounded shadow">
      <h2 className="font-semibold mb-2">Metrics & Evaluation</h2>
      {!metrics && <div className="text-sm text-gray-600">No metrics found.</div>}
      {metrics && (
        <pre className="bg-gray-50 p-3 rounded text-xs overflow-auto">{JSON.stringify(metrics, null, 2)}</pre>
      )}
      <div className="text-xs text-gray-500 mt-2">Metrics are collected from `reports/*.json` and `artifacts/adversarial/eval.csv` (if present).</div>
    </div>
  );
}
