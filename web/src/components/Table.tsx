import React from "react";
import { TableBlock } from "../types";

function toCSV(rows: any[], columns: string[]) {
  const header = columns.join(",");
  const lines = rows.map(r =>
    columns.map(c => JSON.stringify(r[c] ?? "")).join(",")
  );
  return [header, ...lines].join("\n");
}

export default function Table({ block }: { block: TableBlock }) {
  const cols = block.data.columns;
  const rows = block.data.rows;

  const download = () => {
    const blob = new Blob([toCSV(rows, cols)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${block.name}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white rounded-2xl shadow p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold">{block.title || block.name}</h3>
        <button className="text-sm px-3 py-1 rounded bg-gray-100 hover:bg-gray-200" onClick={download}>
          Download CSV
        </button>
      </div>
      <div className="overflow-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              {cols.map(c => (
                <th key={c} className="text-left px-3 py-2 font-medium text-gray-600">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr><td className="px-3 py-4 text-gray-400" colSpan={cols.length}>No data</td></tr>
            ) : (
              rows.map((r, idx) => (
                <tr key={idx} className={idx % 2 ? "bg-gray-50/50" : ""}>
                  {cols.map(c => (
                    <td key={c} className="px-3 py-2 whitespace-nowrap">{r[c] ?? ""}</td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
