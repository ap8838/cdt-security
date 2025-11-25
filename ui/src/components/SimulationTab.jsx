import React, { useState } from "react";
import { generateSamples } from "../api";

// The component now accepts 'datasets' instead of 'assets'
export default function SimulationTab({ datasets = [], onGenerated = () => {} }) {
  const [n, setN] = useState(100);
  // Renamed state from 'asset' to 'dataset' and updated default value logic
  const [dataset, setDataset] = useState(
    datasets && datasets.length ? datasets[0] : "synthetic"
  );
  const [post, setPost] = useState(false);
  const [status, setStatus] = useState(null);

  // Update default dataset if datasets change
  React.useEffect(() => {
    // Check if the current dataset is no longer in the list (or is the default 'synthetic')
    if (datasets && datasets.length && !datasets.includes(dataset) && dataset !== "synthetic") {
      setDataset(datasets[0] || "synthetic");
    } else if (!datasets.length && dataset !== "synthetic") {
      setDataset("synthetic");
    }
  }, [datasets]); // Dependency array updated to use 'datasets'

  const runGen = async () => {
    setStatus("starting");
    try {
      // Use 'dataset' in the payload
      const payload = { n, dataset, post };
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
        {/* Replaced the select to use 'dataset' state and 'datasets' prop */}
        <select value={dataset} onChange={(e) => setDataset(e.target.value)} className="border px-2 py-1 rounded">
          <option value="synthetic">synthetic</option>
          {datasets.map((d) => <option key={d}>{d}</option>)}
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