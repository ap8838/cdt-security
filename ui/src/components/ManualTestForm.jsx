import React, { useState } from "react";
import { postScore } from "../api";

export default function ManualTestForm({ assets = [], dataset }) {
  const [assetId, setAssetId] = useState("");
  const [features, setFeatures] = useState("{}");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const parsed = JSON.parse(features);
      const payload = {
        asset_id: assetId || "manual_test_asset",
        timestamp: new Date().toISOString(),
        features: parsed,
      };
      const res = await postScore(payload, dataset);
      setResult(res);
    } catch (err) {
      setError(`Invalid input or request: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-50 p-3 rounded border">
      <form onSubmit={handleSubmit} className="flex flex-col gap-2">
        <select
          value={assetId}
          onChange={(e) => setAssetId(e.target.value)}
          className="border px-2 py-1 rounded"
        >
          <option value="">-- choose asset --</option>
          {assets.map((a) => (
            <option key={a.asset_id} value={a.asset_id}>
              {a.asset_id}
            </option>
          ))}
        </select>

        <textarea
          rows={3}
          className="border rounded p-2 text-xs font-mono"
          placeholder='{"temperature": 5.4, "humidity": 0.62}'
          value={features}
          onChange={(e) => setFeatures(e.target.value)}
        />

        <button
          type="submit"
          disabled={loading || !dataset}
          className={`px-3 py-1 rounded text-white ${
            dataset ? "bg-blue-600" : "bg-gray-400 cursor-not-allowed"
          }`}
        >
          {loading ? "Scoring..." : dataset ? `Test on ${dataset}` : "Select dataset first"}
        </button>
      </form>

      {error && <div className="text-red-600 text-xs mt-1">{error}</div>}
      {result && (
        <pre className="text-xs bg-white p-2 mt-2 rounded border overflow-auto max-h-40">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}
