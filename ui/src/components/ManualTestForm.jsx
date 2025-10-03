import React, { useEffect, useState } from "react";
import { postScore } from "../api";
import AssetSelector from "./AssetSelector";

export default function ManualTestForm({ assets = [] }) {
  const [asset, setAsset] = useState("");
  const [temp, setTemp] = useState(5.0);
  const [resp, setResp] = useState(null);

  // when assets arrive, pick first asset if not already selected
  useEffect(() => {
    if (assets.length > 0 && !asset) {
      setAsset(assets[0].asset_id);
    }
  }, [assets]);

  const submit = async (e) => {
    e.preventDefault();
    if (!asset) {
      setResp("Please select an asset");
      return;
    }
    const payload = {
      asset_id: asset,
      timestamp: new Date().toISOString(),
      features: { fridge_temperature: Number(temp) },
    };
    try {
      const r = await postScore(payload);
      setResp(JSON.stringify(r, null, 2));
    } catch (err) {
      setResp("error: " + String(err));
    }
  };

  return (
    <form onSubmit={submit} className="flex items-center gap-2">
      <AssetSelector assets={assets} selected={asset} onChange={setAsset} />
      <input
        type="number"
        value={temp}
        onChange={(e) => setTemp(e.target.value)}
        className="border px-2 py-1 rounded w-28"
        step="0.1"
      />
      <button className="bg-blue-600 text-white px-3 py-1 rounded">Send</button>
      <div className="text-xs text-gray-600 ml-2 whitespace-pre-wrap">{resp}</div>
    </form>
  );
}
