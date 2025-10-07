import React from "react";
import DashboardTabs from "./components/DashboardTabs";

export default function App() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">CDT Security — Research Dashboard</h1>
      <DashboardTabs />
    </div>
  );
}
