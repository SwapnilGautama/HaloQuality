import React from "react";

type HeatmapProps = {
  title: string;
  matrix: { rowKeys: string[]; colKeys: string[]; values: number[][] };
  valueLabel?: string;
};

function colorFor(v: number | null, min = 0, max = 100) {
  if (v === null || isNaN(v)) return "transparent";
  const t = Math.max(0, Math.min(1, (v - min) / (max - min || 1)));
  // simple blue scale; customize if needed
  const r = Math.round(240 - 140 * t);
  const g = Math.round(248 - 160 * t);
  const b = 255;
  return `rgb(${r},${g},${b})`;
}

export default function HeatmapTable({ title, matrix, valueLabel = "% within row" }: HeatmapProps) {
  const flat = matrix.values.flat().filter(x => x != null) as number[];
  const min = flat.length ? Math.min(...flat) : 0;
  const max = flat.length ? Math.max(...flat) : 100;

  return (
    <div className="bg-white rounded-2xl shadow p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold">{title}</h3>
        <span className="text-xs text-gray-500">{valueLabel}</span>
      </div>
      <div className="overflow-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr>
              <th className="px-3 py-2 text-left">Row</th>
              {matrix.colKeys.map(c => (
                <th key={c} className="px-3 py-2 text-left">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.rowKeys.map((rk, i) => (
              <tr key={rk} className={i % 2 ? "bg-gray-50/50" : ""}>
                <td className="px-3 py-2 font-medium">{rk}</td>
                {matrix.colKeys.map((ck, j) => {
                  const v = matrix.values[i][j];
                  return (
                    <td key={ck} className="px-3 py-2" style={{ background: colorFor(v, min, max) }}>
                      {v ?? ""}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
