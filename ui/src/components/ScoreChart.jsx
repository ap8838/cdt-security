import React from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";

export default function ScoreChart({ data = [], showSynthetic = false }) {
  // prepare two series: real and synthetic
  // Use slice(-200) to get the latest 200 items, map them, and reverse to show oldest on left
  const real = data.filter(d => !d.synthetic).slice(-200).map(d => ({
    id: d.id,
    ts: d.timestamp,
    score: Number(d.score)
  })).reverse();

  const synth = data.filter(d => d.synthetic).slice(-200).map(d => ({
    id: d.id,
    ts: d.timestamp,
    score: Number(d.score)
  })).reverse();

  // Align by index: we'll render both lines on same chart; both use their own data
  return (
    <div style={{ height: 360 }}>
      <ResponsiveContainer width="100%" height="100%">
        {/* We use the <LineChart> without a main data prop */}
        <LineChart>
          <XAxis dataKey="ts" hide />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip />

          {/* Real Alerts Line (Blue) */}
          <Line data={real} type="monotone" dataKey="score" stroke="#2563eb" dot={false} />

          {/* Synthetic Alerts Line (Orange, Dashed) - Rendered if shown or if there is synthetic data */}
          { (showSynthetic || synth.length > 0) && (
            <Line
              data={synth}
              type="monotone"
              dataKey="score"
              stroke="#f59e0b"
              dot={false}
              strokeDasharray="4 4"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}