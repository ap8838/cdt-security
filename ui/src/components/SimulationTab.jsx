import React, { useState } from "react";
import { generateSamples } from "../api";

export default function SimulationTab({ assets = [], onGenerated = () => {} }) {
  const [n, setN] = useState(100);
  const [asset, setAsset] = useState(assets && assets.length ? assets[0].asset_id : "synthetic");
  const [post, setPost] = useState(false);
  const [status, setStatus] = useState(null);

  // update default asset if assets change
  React.useEffect(() => {
    if (assets && assets.length && !assets.find(a => a.asset_id === asset)) {
      setAsset(assets[0]?.asset_id || "synthetic");
    }
  }, [assets]);

  const runGen = async () => {
    setStatus("starting");
    try {
      const payload = { n, asset, post };
      const res = await generateSamples(payload);
      setStatus(`started: ${res.started ? "ok" : "failed"}`);
      onGenerated();
    } catch (err) {
      setStatus("error: " + String(err));
    }
  };

  return (
    <div className="bg-white p-4 rounded shadow">
      <h2 className="font-semibold mb-2">Simulation — generate synthetic attacks (cGAN)</h2>

      <div className="flex items-center gap-2 mb-3">
        <select value={asset} onChange={(e) => setAsset(e.target.value)} className="border px-2 py-1 rounded">
          <option value="synthetic">synthetic</option>
          {assets.map((a) => <option key={a.asset_id} value={a.asset_id}>{a.asset_id}</option>)}
        </select>

        <input type="number" className="border px-2 py-1 rounded w-28" value={n} onChange={(e) => setN(Number(e.target.value))} />

        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={post} onChange={(e) => setPost(e.target.checked)} />
          Post to API
        </label>

        <button className="bg-blue-600 text-white px-3 py-1 rounded" onClick={runGen}>Generate</button>
      </div>

      {status && <div className="text-sm text-gray-700">Status: {status}</div>}
      <div className="text-xs text-gray-500 mt-2">The server will run generation in background. If you enabled Post, generated events are sent to the /score endpoint.</div>
    </div>
  );
}
