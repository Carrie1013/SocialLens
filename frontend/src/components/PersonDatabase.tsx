import React, { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8080";

export const PersonDatabase: React.FC = () => {
  const [persons, setPersons] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isAdding, setIsAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchPersons = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/face-db`);
      if (!res.ok) return;
      const data = await res.json();
      setPersons(data.persons ?? []);
    } catch {}
  }, []);

  useEffect(() => {
    fetchPersons();
  }, [fetchPersons]);

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !file) return;
    setIsAdding(true);
    setError(null);
    setSuccess(null);
    try {
      const form = new FormData();
      form.append("name", name.trim());
      form.append("file", file);
      const res = await fetch(`${API_BASE}/api/face-db?name=${encodeURIComponent(name.trim())}`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setSuccess(`Registered "${name.trim()}" successfully.`);
      setName("");
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      await fetchPersons();
    } catch (err: any) {
      setError(err.message || "Registration failed");
    } finally {
      setIsAdding(false);
    }
  };

  const handleRemove = async (personName: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/face-db/${encodeURIComponent(personName)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? `HTTP ${res.status}`);
      }
      await fetchPersons();
    } catch (err: any) {
      setError(err.message || "Remove failed");
    }
  };

  return (
    <div className="glass-panel rounded-xl p-4 border border-gray-800">
      <div className="text-xs font-mono text-gray-500 uppercase mb-3">People Database</div>

      {/* Registered people list */}
      {persons.length > 0 ? (
        <ul className="flex flex-col gap-1 mb-3">
          {persons.map((p) => (
            <li key={p} className="flex items-center justify-between bg-gray-800/50 rounded px-2 py-1">
              <span className="text-sm font-mono text-green-400">{p}</span>
              <button
                onClick={() => handleRemove(p)}
                className="text-xs text-gray-500 hover:text-red-400 transition-colors ml-2"
                title={`Remove ${p}`}
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-xs text-gray-600 font-mono mb-3">
          No people registered yet.
        </p>
      )}

      {/* Add person form */}
      <form onSubmit={handleAdd} className="flex flex-col gap-2">
        <input
          type="text"
          placeholder="Name (e.g. Alice)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm font-mono
                     text-gray-200 placeholder-gray-600 focus:outline-none focus:border-green-600"
        />
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="text-xs text-gray-400 file:mr-2 file:py-1 file:px-2 file:rounded
                     file:border-0 file:text-xs file:bg-gray-800 file:text-gray-300
                     hover:file:bg-gray-700 cursor-pointer"
        />
        <button
          type="submit"
          disabled={isAdding || !name.trim() || !file}
          className="text-xs py-1 rounded border border-gray-700 hover:border-green-500
                     text-gray-400 hover:text-green-400 transition-colors disabled:opacity-40"
        >
          {isAdding ? "Registering…" : "+ Add Person"}
        </button>
      </form>

      {error && (
        <div className="mt-2 text-xs text-red-400 font-mono">{error}</div>
      )}
      {success && (
        <div className="mt-2 text-xs text-green-400 font-mono">{success}</div>
      )}
    </div>
  );
};
