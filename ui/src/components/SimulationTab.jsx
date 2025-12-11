import React, { useEffect, useState } from "react";
import { generateSamples, fetchAdversarialModels } from "../api";

export default function SimulationTab({ onGenerated = () => {} }) {
  const [n, setN] = useState(50);
  const [dataset, setDataset] = useState("");
  const [postToApi, setPostToApi] = useState(true);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState("");
  const [models, setModels] = useState([]);

  useEffect(() => {
    // load available cGAN models
    fetchAdversarialModels()
      .then((m) => {
        setModels(m || []);
        if ((m || []).length && !dataset) setDataset(m[0]);
      })
      .catch((err) => {
        console.error("Failed to load adversarial models:", err);
        setMessage("Failed to load adversarial models. Check backend.");
      });
  }, []);

  async function handleGenerate() {
    if (!dataset) {
      setMessage("⚠️ Select a target dataset (cGAN model).");
      return;
    }
    if (!Number.isInteger(n) || n <= 0) {
      setMessage("⚠️ Enter a positive integer for Count.");
      return;
    }

    setRunning(true);
    setMessage("Generating synthetic samples...");

    try {
      const resp = await generateSamples({ dataset, n, post: postToApi });
      if (resp && resp.error) {
        setMessage("Error: " + resp.error);
      } else {
        setMessage(
          `Done. Generated=${resp.generated ?? n} Posted=${resp.posted ?? 0} Anomalies=${resp.anomalies_detected ?? 0}`
        );
        // Notify parent and provide the dataset that was generated for
        // so the dashboard can switch the dataset filter to show results.
        onGenerated(resp.dataset || dataset);
      }
    } catch (err) {
      console.error("generateSamples error:", err);
      const errText = err?.response?.data || err?.message || String(err);
      setMessage("Request failed: " + JSON.stringify(errText));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="bg-white p-4 rounded shadow">
      <h2 className="font-semibold mb-4">cGAN Simulation — Synthetic Attack Generator</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
        <div>
          <label className="block text-sm font-medium mb-1">Target GAN (dataset)</label>
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="w-full border px-2 py-1 rounded"
          >
            <option value="">-- select target GAN --</option>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Count</label>
          <input
            type="number"
            value={n}
            min={1}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-full border px-2 py-1 rounded"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Post to API</label>
          <div className="flex items-center gap-2">
            <input
              id="post-sim"
              type="checkbox"
              checked={postToApi}
              onChange={(e) => setPostToApi(e.target.checked)}
            />
            <label htmlFor="post-sim" className="text-sm">Send generated events to /score</label>
          </div>
        </div>
      </div>

      <div className="mt-4">
        <button
          onClick={handleGenerate}
          disabled={running}
          className={`px-3 py-2 rounded ${running ? "bg-gray-300" : "bg-blue-600 text-white"}`}
        >
          {running ? "Generating..." : "Generate Samples"}
        </button>
      </div>

      <div className="mt-3 text-sm text-gray-700">{message}</div>

      <div className="mt-4 text-xs text-gray-500">
        Each option corresponds to a trained cGAN file in <code>artifacts/adversarial/{`{dataset}_cgan.pt`}</code>.
        Selecting one will generate samples from that specific cGAN and (optionally) POST them to the score endpoint for that dataset.
      </div>
    </div>
  );
}