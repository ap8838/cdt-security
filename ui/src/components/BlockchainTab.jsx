import React, { useState } from "react";
import { fetchBlockRecord } from "../api";

export default function BlockchainTab({ alerts = [] }) {
  const [tx, setTx] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const recentTxs = alerts.map(a => a.tx_hash).filter(Boolean);

  const verify = async (txHash) => {
    if (!txHash) return setError("No tx_hash provided");
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetchBlockRecord(txHash);
      setResult(res);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-4 rounded shadow">
      <h2 className="font-semibold mb-2">Blockchain — Alert verification</h2>

      <div className="flex gap-2 items-center mb-3">
        <select
          className="border px-2 py-1 rounded"
          value={tx}
          onChange={(e) => setTx(e.target.value)}
        >
          <option value="">-- choose tx_hash from alerts --</option>
          {recentTxs.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>

        <input
          className="border px-2 py-1 rounded w-96"
          placeholder="or paste tx_hash here (0x...)"
          value={tx}
          onChange={(e) => setTx(e.target.value)}
        />

        <button
          className="bg-blue-600 text-white px-3 py-1 rounded"
          onClick={() => verify(tx)}
          disabled={loading}
        >
          {loading ? "Verifying..." : "Verify"}
        </button>
      </div>

      {error && <div className="text-red-600 mb-2">{error}</div>}

      {result && (
        <div className="text-sm">
          <div className="mb-2">Verified on-chain record:</div>
          <pre className="bg-gray-50 p-3 rounded text-xs overflow-auto">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}

      <div className="mt-4 text-xs text-gray-600">
        Note: Ganache is a local chain. To inspect tx you can open Ganache UI (if running) or use the verify endpoint above.
      </div>
    </div>
  );
}
