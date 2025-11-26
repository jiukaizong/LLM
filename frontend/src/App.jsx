import { BrowserRouter, Routes, Route, NavLink, Navigate } from "react-router-dom";
import Employee from "./pages/Employee";
import Credit from "./pages/Credit";
import Insights from "./pages/Insights";
import Chat from "./pages/Chat";

function NavBtn({ to, children }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-4 py-2 rounded-2xl shadow-sm ${isActive ? "bg-gray-200" : "bg-white border"}`
      }
    >
      {children}
    </NavLink>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 p-6 space-y-6">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">LLM Risk & Performance</h1>
          <div className="flex gap-2">
            <NavBtn to="/employee">Employee</NavBtn>
            <NavBtn to="/credit">Credit</NavBtn>
            <NavBtn to="/insights">Insights</NavBtn>
            <NavBtn to="/chat">Chat</NavBtn> 
          </div>
        </header>

        <main className="bg-white rounded-2xl shadow p-6">
          <Routes>
            <Route path="/" element={<Navigate to="/employee" replace />} />
            <Route path="/employee" element={<Employee />} />
            <Route path="/credit" element={<Credit />} />
            <Route path="/insights" element={<Insights />} />
            <Route path="/chat" element={<Chat />} /> 
            <Route path="*" element={<Navigate to="/employee" replace />} />
          </Routes>
        </main>

        <footer className="text-xs text-gray-500">© 2025 LLM Risk & Performance</footer>
      </div>
    </BrowserRouter>
  );
}
