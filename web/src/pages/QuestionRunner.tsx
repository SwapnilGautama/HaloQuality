import React, { useEffect, useMemo, useState } from "react";
import { listQuestions, runQuestion } from "../api/client";
import { QuestionResponse, TableBlock, ChartSpec } from "../types";
import MetricCard from "../components/MetricCard";
import Table from "../components/Table";
import BarChart from "../components/BarChart";
import HeatmapTable from "../components/HeatmapTable";

export default function QuestionRunner() {
  const [apiError, setApiError] = useState<string | null>(null);
  const [questions, setQuestions] = useState<string[]>([]);
  const [qId, setQId] = useState<string>("complaint_analysis");
  const [month, setMonth] = useState<string>("2025-06");
  const [groupBy, setGroupBy] = useState<string>("Portfolio_std");
  const [payload, setPayload] = useState<QuestionResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    listQuestions()
      .then(qs => {
        setQuestions(qs);
        if (qs.includes("complaint_analysis")) setQId("complaint_analysis");
        else if (qs[0]) setQId(qs[0]);
      })
      .catch(e => setApiError(String(e)));
  }, []);

  const run = async () => {
    setApiError(null);
    setLoading(true);
    try {
      const data = await runQuestion(qId, month, groupBy);
      setPayload(data);
    } catch (e: any) {
      setApiError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  // helpers
  const tableByName = (name: string) =>
    payload?.tables.find(t => t.name === name) as TableBlock | undefined;

  const chartByName = (name: string) =>
    payload?.charts.find(c => c.name === name) as ChartSpec | undefined;

  const dataRef = (id?: string) =>
    (id && payload?.dataRefs?.[id]) ? payload!.dataRefs[id] : [];

  // heatmap build
  const heatmapMatrix = useMemo(() => {
    const t = tableByName("reasons_heatmap");
    if (!t) return null;
    const cols = t.data.columns;
    const rows = t.data.rows;
    const metricCols = new Set(["Reason","Count","Value","Prev_Count","Prev_Value","Delta","Row_Total","Col_Total","Grand_Total"]);
    const rowDim = cols.find(c => !metricCols.has(c) && c !== "Reason");
    if (!rowDim) return null;
    const colKeys = Array.from(new Set(rows.map(r => r["Reason"]))).filter(Boolean) as string[];
    const rowKeys = Array.from(new Set(rows.map(r => r[rowDim]))).filter(Boolean) as string[];
    const grid: (number | null)[][] = rowKeys.map(() => colKeys.map(() => null));
    const index = new Map<string, number>();
    colKeys.forEach((k, i) => index.set(k, i));
    const rIndex = new Map<string, number>();
    rowKeys.forEach((k, i) => rIndex.set(k, i));
    rows.forEach(r => {
      const i = rIndex.get(r[rowDim]);
      const j = index.get(r["Reason"]);
      if (i !== undefined && j !== undefined) {
        grid[i][j] = typeof r["Value"] === "number" ? Math.round((r["Value"] + Number.EPSILON) * 10) / 10 : Number(r["Value"]) || null;
      }
    });
    return { rowDim, matrix: { rowKeys, colKeys, values: grid } };
  }, [payload]);

  return (
    <div className="max-w-[1400px] mx-auto px-6 py-8">
      <div className="flex items-end gap-4 mb-6">
        <div>
          <label className="block text-xs text-gray-500">Question</label>
          <select className="border rounded-lg px-3 py-2" value={qId} onChange={e => setQId(e.target.value)}>
            {questions.map(q => <option key={q} value={q}>{q}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500">Month (YYYY-MM)</label>
          <input className="border rounded-lg px-3 py-2" value={month} onChange={e => setMonth(e.target.value)} />
        </div>
        <div className="grow">
          <label className="block text-xs text-gray-500">group_by (CSV)</label>
          <input className="border rounded-lg w-full px-3 py-2" value={groupBy} onChange={e => setGroupBy(e.target.value)} />
        </div>
        <button onClick={run} className="rounded-xl bg-black text-white px-5 py-2.5 hover:opacity-90">
          {loading ? "Running…" : "Run"}
        </button>
      </div>

      {apiError && <div className="bg-red-50 text-red-700 p-3 rounded-xl mb-4">{apiError}</div>}

      {payload && (
        <>
          <div className="bg-white rounded-2xl shadow p-4 mb-6">
            <h2 className="text-lg font-semibold mb-2">Insights</h2>
            <p className="text-gray-700 whitespace-pre-line">{payload.insights || "—"}</p>
          </div>

          {/* Cards */}
          {payload.cards?.[0]?.data && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <MetricCard label="Complaints / 1k" value={payload.cards[0].data["rate"]} delta={payload.cards[0].data["rate_delta"]}/>
              <MetricCard label="Complaints" value={payload.cards[0].data["complaints"]} delta={payload.cards[0].data["complaints_delta"]}/>
              <MetricCard label="Unique Cases" value={payload.cards[0].data["cases"]} delta={payload.cards[0].data["cases_delta"]}/>
              <MetricCard label="NPS" value={payload.cards[0].data["nps"]} delta={payload.cards[0].data["nps_delta"]}/>
            </div>
          )}

          {/* Drivers bar */}
          {(() => {
            const spec = chartByName("drivers_bar");
            if (!spec) return null;
            const data = dataRef(spec.dataRef);
            return <div className="mb-6">
              <BarChart title="Top drivers (Rate Δ)" data={data} x={spec.x} y={spec.y} sort={spec.sort}/>
            </div>;
          })()}

          {/* Tables */}
          <div className="grid grid-cols-1 gap-6">
            {payload.tables.map(t => <Table key={t.name} block={t}/>)}
          </div>

          {/* Reasons heatmap (pretty) */}
          {heatmapMatrix && (
            <div className="mt-6">
              <HeatmapTable
                title={`Reasons × ${heatmapMatrix.rowDim} (Row %)`}
                matrix={heatmapMatrix.matrix}
              />
            </div>
          )}
        </>
      )}

      {!payload && <div className="text-gray-500">Run a question to see results.</div>}
    </div>
  );
}
