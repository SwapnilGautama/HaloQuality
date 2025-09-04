import React from "react";
import QuestionRunner from "./pages/QuestionRunner";

export default function App() {
  return (
    <div>
      <header className="bg-white border-b">
        <div className="max-w-[1400px] mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold">Halo Quality</h1>
          <div className="text-xs text-gray-500">React UI · Vite + Tailwind + Recharts</div>
        </div>
      </header>
      <QuestionRunner />
    </div>
  );
}
