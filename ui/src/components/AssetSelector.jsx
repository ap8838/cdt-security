import React from "react";

export default function AssetSelector({ assets = [], selected, onChange }) {
  return (
    <select
      value={selected || ""}
      onChange={(e) => onChange(e.target.value || null)}
      className="border px-2 py-1 rounded"
    >
      <option value="">-- all assets --</option>
      {assets.map((a) => (
        <option key={a.asset_id} value={a.asset_id}>
          {a.asset_id} {a.last_ts ? `(${a.last_ts})` : ""}
        </option>
      ))}
    </select>
  );
}
