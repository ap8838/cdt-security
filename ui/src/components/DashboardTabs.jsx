import React, { useEffect, useState, useRef } from "react";
import { fetchAlerts, fetchDatasets } from "../api";
import AlertsTable from "./AlertsTable";
import ScoreChart from "./ScoreChart";
import BlockchainTab from "./BlockchainTab";
import SimulationTab from "./SimulationTab";
import MetricsTab from "./MetricsTab";

export default function DashboardTabs() {
  const [tab, setTab] = useState("overview");
  const [alerts, setAlerts] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [dataset, setDataset] = useState("");
  const pollingRef = useRef(null);

  useEffect(() => {
    loadDatasets();
    loadAlerts();
    pollingRef.current = setInterval(() => loadAlerts(), 2000);
    return () => clearInterval(pollingRef.current);
  }, []);

  // when dataset changes, reload alerts immediately
  useEffect(() => {
    loadAlerts();
  }, [dataset]);

  function loadDatasets() {
    fetchDatasets().then(setDatasets).catch(console.error);
  }

  function loadAlerts() {
    // request backend alerts filtered by dataset (backend handles empty dataset as "all")
    fetchAlerts(200, 0, dataset || "")
      .then((rows = []) => {
        setAlerts(rows);
      })
      .catch(console.error);
  }

  // Build the filtered data for chart — FILTER ONLY BY DATASET (as requested)
  let filtered = alerts;
  if (dataset) {
    // robust matching: accept equality, startsWith, or contains
    const ds = String(dataset);
    filtered = alerts.filter((a) => {
      const id = String(a.asset_id || "");
      if (id === ds) return true;
      if (id.startsWith(ds)) return true;
      if (id.includes(ds)) return true;
      // try replacing dashes/underscores and check
      const normalized = id.replace(/[-_]/g, "");
      const normDs = ds.replace(/[-_]/g, "");
      if (normalized.startsWith(normDs) || normalized.includes(normDs)) return true;
      return false;
    });
  }

  return (
    <div className="container mx-auto p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">CDT Security — Dashboard</h1>
        <div className="flex gap-2">
          {["overview", "blockchain", "simulation", "metrics"].map((t) => (
            <button
              key={t}
              className={`px-3 py-1 rounded ${tab === t ? "bg-blue-600 text-white" : "bg-gray-100"}`}
              onClick={() => setTab(t)}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {tab === "overview" && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-2 bg-white p-4 rounded shadow">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <select
                  value={dataset || ""}
                  onChange={(e) => setDataset(e.target.value || "")}
                  className="border px-2 py-1 rounded"
                >
                  <option value="">-- select dataset (show all) --</option>
                  {datasets.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>

                <div className="text-sm text-gray-600">
                  Showing <strong>{filtered.length}</strong> alerts (graph)
                </div>
              </div>
            </div>

            <ScoreChart data={filtered} />
          </div>

          <div className="bg-white p-4 rounded shadow">
            <h2 className="font-semibold mb-2">Latest Alerts</h2>
            {/* keep table showing the latest alerts returned (dataset-scoped by our fetch) */}
            <AlertsTable rows={alerts.slice(0, 50)} />
          </div>
        </div>
      )}

      {tab === "blockchain" && <BlockchainTab alerts={alerts} />}

      {tab === "simulation" && (
        <SimulationTab
          onGenerated={(generatedDataset) => {
            if (generatedDataset) {
              // switch dashboard filter to show this dataset
              setDataset(generatedDataset);
            }
            // reload alerts (will fetch with dataset filter)
            loadAlerts();
          }}
        />
      )}
      {tab === "metrics" && <MetricsTab />}
    </div>
  );
}
