"use client";
import { useEffect, useState } from "react";
import { fetchEvents, lineOf } from "./useEvents";

export default function EventList({ limit = 100, pollMs = 5000, debug = false }) {
  const [items, setItems] = useState([]);
  const [status, setStatus] = useState("idle");
  const [err, setErr] = useState(null);
  const [meta, setMeta] = useState(null); // 디버그용

  async function load() {
    try {
      setStatus("loading");
      const data = await fetchEvents({ limit });
      setItems(data);
      setMeta({ count: data?.length ?? 0, sample: data?.[0] ?? null });
      setErr(null);
      setStatus("ok");
    } catch (e) {
      setStatus("error");
      setErr(e?.status === 401 ? "인증 만료: 다시 로그인 해주세요." : (e?.message || String(e)));
    }
  }

  useEffect(() => {
    const t = setTimeout(load, 100);
    return () => clearTimeout(t);
  }, [limit]);

  useEffect(() => {
    if (pollMs > 0) {
      const t = setInterval(load, pollMs);
      return () => clearInterval(t);
    }
  }, [pollMs]);

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold">이벤트 로그</h3>
        <button onClick={load} className="px-3 py-1 rounded-lg bg-gray-800 text-white">새로고침</button>
      </div>

      {err && <p className="text-sm text-red-500">{err}</p>}
      {debug && (
        <p className="text-xs text-gray-500 mb-2">
          count:{meta?.count ?? 0} {meta?.sample ? `· sample keys: ${Object.keys(meta.sample).join(", ")}` : ""}
        </p>
      )}

      <ul className="space-y-2 text-sm">
        {items.map((ev) => (
          <li key={ev.id ?? `${ev.created_at}-${ev.seat_id}-${ev.camera_id}-${Math.random()}`}
              className="bg-black text-white px-4 py-2 rounded-lg shadow-sm border border-gray-700">
            {lineOf(ev)}
          </li>
        ))}
      </ul>

      {items.length === 0 && !err && <p className="text-sm text-gray-600">표시할 로그가 없습니다.</p>}
    </div>
  );
}
