import axios from "axios";

// Use /api by default (nginx proxy). You can still override with VITE_API_BASE.
const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000
});

export async function listQuestions(): Promise<string[]> {
  const { data } = await api.get("/question/list");
  return data?.questions ?? [];
}

export async function runQuestion(questionId: string, month: string, groupByCSV: string) {
  const { data } = await api.get(`/question/${encodeURIComponent(questionId)}`, {
    params: { month, group_by: groupByCSV }
  });
  return data;
}
