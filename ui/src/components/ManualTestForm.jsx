import React, { useState } from "react";
import { postScore } from "../api";

export default function ManualTestForm() {
  const [asset, setAsset] = useState("iot_fridge");
  const [temp, setTemp] = useState(5.0);
  const [resp, setResp] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
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
      <input value={asset} onChange={(e) => setAsset(e.target.value)} className="border px-2 py-1 rounded w-36" />
      <input type="number" value={temp} onChange={(e) => setTemp(e.target.value)} className="border px-2 py-1 rounded w-28" step="0.1" />
      <button className="bg-blue-600 text-white px-3 py-1 rounded">Send</button>
      <div className="text-xs text-gray-600 ml-2 whitespace-pre-wrap">{resp}</div>
    </form>
  );
}
