import React from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";

export default function ScoreChart({ data = [] }) {
  // keep last 200 and convert timestamp -> readable
  const chartData = data.slice(0, 200).map(d => ({
    id: d.id,
    ts: d.timestamp,
    score: Number(d.score),
  })).reverse();

  return (
    <div style={{ height: 360 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <XAxis dataKey="ts" hide />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip />
          <Line type="monotone" dataKey="score" stroke="#2563eb" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
