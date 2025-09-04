import React from "react";
import {
  ResponsiveContainer,
  BarChart as RBarChart,
  XAxis,
  YAxis,
  Tooltip,
  Bar,
  CartesianGrid
} from "recharts";

type Props = {
  title: string;
  data: Record<string, any>[];
  x: string;
  y: string;
  sort?: string;
};

export default function BarChart({ title, data, x, y, sort }: Props) {
  let d = data.filter(row => row[x] !== undefined && row[y] !== undefined);
  if (sort && sort.toLowerCase().startsWith("desc")) {
    d = [...d].sort((a, b) => (b[y] ?? 0) - (a[y] ?? 0));
  }

  return (
    <div className="bg-white rounded-2xl shadow p-4">
      <h3 className="font-semibold mb-3">{title}</h3>
      <div className="h-[420px]">
        <ResponsiveContainer>
          <RBarChart data={d}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={x} hide={d.length > 40}/>
            <YAxis />
            <Tooltip />
            <Bar dataKey={y} />
          </RBarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
