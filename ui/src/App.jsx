import React, { useEffect, useState, useRef } from "react";
import { fetchAlerts, fetchAssets } from "./api";
import AlertsTable from "./components/AlertsTable";
import ScoreChart from "./components/ScoreChart";
import AssetSelector from "./components/AssetSelector";
import ManualTestForm from "./components/ManualTestForm";

export default function App() {
  const [alerts, setAlerts] = useState([]);
  const [assets, setAssets] = useState([]);
  const [asset, setAsset] = useState(null);
  const pollingRef = useRef(null);

  useEffect(() => {
    // initial load assets
    fetchAssets().then(setAssets).catch(console.error);
    loadAlerts();

    // poll alerts every 2s
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

  const filtered = asset ? alerts.filter(a => a.asset_id === asset) : alerts;

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">CDT Security — Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-2 bg-white p-4 rounded shadow">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <AssetSelector
                assets={assets}
                selected={asset}
                onChange={setAsset}
              />
              <div className="text-sm text-gray-600">
                Showing <strong>{filtered.length}</strong> alerts
              </div>
            </div>
            <ManualTestForm />
          </div>

          <ScoreChart data={filtered} />
        </div>

        <div className="bg-white p-4 rounded shadow">
          <h2 className="font-semibold mb-2">Latest Alerts</h2>
          <AlertsTable rows={alerts.slice(0, 50)} />
        </div>
      </div>
    </div>
  );
}
