import React from "react";

export default function AlertsTable({ rows = [] }) {
  return (
    <div className="overflow-auto max-h-[480px]">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-100 sticky top-0">
          <tr>
            <th className="p-2">id</th>
            <th className="p-2">asset</th>
            <th className="p-2">ts</th>
            <th className="p-2">score</th>
            <th className="p-2">model</th>
            <th className="p-2">threshold</th>
            <th className="p-2">anomaly</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-b">
              <td className="p-2">{r.id}</td>
              <td className="p-2">{r.asset_id}</td>
              <td className="p-2">{r.timestamp}</td>
              <td className="p-2">{Number(r.score).toFixed(3)}</td>
              <td className="p-2">{r.model}</td>
              <td className="p-2">{Number(r.threshold).toFixed(4)}</td>
              <td className={`p-2 font-semibold ${r.is_anomaly ? "text-red-600" : "text-green-600"}`}>
                {r.is_anomaly ? "YES" : "no"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
