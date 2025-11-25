import React, { useEffect, useState, useRef } from "react";
import { fetchAlerts, fetchAssets, fetchDatasets } from "../api";
import AlertsTable from "./AlertsTable";
import ScoreChart from "./ScoreChart";
import BlockchainTab from "./BlockchainTab";
import SimulationTab from "./SimulationTab";
import MetricsTab from "./MetricsTab";

export default function DashboardTabs() {
  const [tab, setTab] = useState("overview");
  const [alerts, setAlerts] = useState([]);
  const [assets, setAssets] = useState([]);
  const [asset, setAsset] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [dataset, setDataset] = useState("");
  const pollingRef = useRef(null);

  useEffect(() => {
    loadDatasets();
    loadAssets();
    loadAlerts();

    pollingRef.current = setInterval(loadAlerts, 2000);
    return () => clearInterval(pollingRef.current);
  }, []);

  function loadAlerts() {
    fetchAlerts(200, 0)
      .then((rows) => {
        setAlerts(rows);
        if (!asset && rows.length) {
          setAsset(rows[0].asset_id);
        }
      })
      .catch(console.error);
  }

  function loadAssets() {
    fetchAssets().then(setAssets).catch(console.error);
  }

  function loadDatasets() {
    fetchDatasets().then(setDatasets).catch(console.error);
  }

  const filtered = asset ? alerts.filter((a) => a.asset_id === asset) : alerts;

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
                  <option value="">-- select dataset --</option>
                  {datasets.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
                <div className="text-sm text-gray-600">
                  Showing <strong>{filtered.length}</strong> alerts
                </div>
              </div>
            </div>

            <ScoreChart data={filtered} />
          </div>

          <div className="bg-white p-4 rounded shadow">
            <h2 className="font-semibold mb-2">Latest Alerts</h2>
            <AlertsTable rows={alerts.slice(0, 50)} />
          </div>
        </div>
      )}

      {tab === "blockchain" && <BlockchainTab alerts={alerts} />}

      {tab === "simulation" && (
        <SimulationTab
          assets={assets}
          onGenerated={() => {
            loadAlerts();
            loadAssets();
          }}
        />
      )}

      {tab === "metrics" && <MetricsTab />}
    </div>
  );
}
