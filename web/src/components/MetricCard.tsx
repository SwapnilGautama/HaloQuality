import React from "react";

type Props = {
  label: string;
  value?: number | string | null;
  delta?: number | string | null;
};

export default function MetricCard({ label, value, delta }: Props) {
  const deltaStr =
    delta === null || delta === undefined || delta === "" ? "—" : `${delta}`;
  const up = typeof delta === "number" && delta > 0;
  const down = typeof delta === "number" && delta < 0;

  return (
    <div className="bg-white rounded-2xl shadow p-4 flex flex-col gap-2 min-h-[112px]">
      <div className="text-gray-500 text-sm">{label}</div>
      <div className="flex items-end gap-3">
        <div className="text-3xl font-semibold text-gray-900">
          {value ?? "—"}
        </div>
        <div
          className={`text-sm ${up ? "text-green-600" : down ? "text-red-600" : "text-gray-400"}`}
        >
          {deltaStr}
        </div>
      </div>
    </div>
  );
}
