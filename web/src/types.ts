export type TableBlock = {
  name: string;
  title: string;
  data: { columns: string[]; rows: Record<string, any>[] };
};

export type ChartSpec = {
  name: string;
  type: "bar";
  x: string;
  y: string;
  dataRef: string;
  sort?: "asc" | "desc" | string;
};

export type QuestionResponse = {
  id: string;
  version: number;
  params: { month: string; group_by: string[] };
  insights: string;
  cards: { name: string; title: string; data: Record<string, number | string | null> }[];
  tables: TableBlock[];
  charts: ChartSpec[];
  dataRefs: Record<string, Record<string, any>[]>;
};
