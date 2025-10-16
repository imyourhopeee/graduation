"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchEvents } from "./useEvents";
import { lineOf } from "./eventFormat";

// ---- utils ----
function normType(ev) {
  const t = String(ev.type ?? ev.event_type ?? "").toLowerCase();
  return t || "event";
}
function stableKey(ev) {
  // 서버 id가 최우선. 없으면 시간+seat+cam 조합으로 안정 키
  const ts = ev.ended_at ?? ev.at ?? ev.started_at ?? ev.created_at ?? "";
  const seat = ev.seat_id ?? "-";
  const cam = ev.camera_id ?? "-";
  return String(ev.id ?? `${ts}|${seat}|${cam}`);
}
function arraysEqualByKey(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (stableKey(a[i]) !== stableKey(b[i])) return false;
  }
  return true;
}

const TYPE_LABELS = {
  intrusion: "자리 침입",
  intrusion_started: "침입 시작",
  intrusion_ended: "자리 복귀",
  security: "보안 탐지",
  system_start: "시스템 시작",
};
const toKo = (t) => TYPE_LABELS[t] ?? t;

export default function FilterableEventList({
  limit = 100,
  pollMs = 5000,
  placeholder = "검색어 입력 (타입, 사용자, 좌석, 카메라 등)",
}) {
  const [items, setItems] = useState([]);
  const [err, setErr] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  // 검색/필터 상태
  const [q, setQ] = useState("");
  const [selType, setSelType] = useState("all");

  const listRef = useRef(null);

  async function load() {
    try {
      setRefreshing(true);       // 기존 리스트 유지
      const data = await fetchEvents({ limit }); // 최신순 정렬된 배열
      // 동일하면 set 생략 → 리렌더 방지
      if (!arraysEqualByKey(items, data)) setItems(data);
      setErr(null);
    } catch (e) {
      setErr(e?.status === 401 ? "인증 만료: 다시 로그인 해주세요." : (e?.message || String(e)));
    } finally {
      setRefreshing(false);
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
  }, [pollMs, items]); // items가 바뀌어야 동기화 주기 유지

  // 드롭다운 옵션
  const typeOptions = useMemo(() => {
    const set = new Set(items.map(normType).filter(Boolean));
    ["intrusion", "intrusion_started", "intrusion_ended", "security", "system_start"].forEach((t) => set.add(t));
    return ["all", ...Array.from(set)];
  }, [items]);

  // 필터링
  const filtered = useMemo(() => {
    const qlower = q.trim().toLowerCase();
    return items.filter((ev) => {
      const t = normType(ev);
      const passType = selType === "all" || t === selType;
      if (!qlower) return passType;

      const blob = [
        lineOf(ev),
        t,
        ev.person_id, ev.identity, ev.meta?.user_label,
        String(ev.seat_id), String(ev.camera_id), ev.meta?.device_id,
      ].filter(Boolean).join(" ").toLowerCase();

      return passType && blob.includes(qlower);
    });
  }, [items, q, selType]);

  return (
    <div className="w-full relative">
      {/* 상단 바 */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">이벤트 로그</h3>
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={placeholder}
            className="w-[48ch] max-w-[60vw] px-3 py-2 border border-gray-300 rounded-lg shadow-sm"
          />
          <select
            value={selType}
            onChange={(e) => setSelType(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg shadow-sm"
          >
            {typeOptions.map((t) => (
              <option key={t} value={t}>
                {t === "all" ? "전체" : toKo(t)}
              </option>
            ))}
          </select>
          <button
            onClick={load}
            className="px-3 py-2 rounded-lg bg-gray-800 text-white disabled:opacity-60"
            disabled={refreshing}
            title="새로고침"
          >
            {refreshing ? "갱신 중…" : "새로고침"}
          </button>
        </div>
      </div>

      {/* 리스트 (항상 마운트 유지) */}
      <ul ref={listRef} className="space-y-2 text-sm min-h-[2rem]">
        {filtered.length === 0 ? (
          <p className="text-gray-600">표시할 로그가 없습니다.</p>
        ) : (
          filtered.map((ev) => (
            <li
              key={stableKey(ev)}   // ← 안정 키
              className="bg-black text-white px-4 py-2 rounded-lg shadow-sm border border-gray-700"
            >
              {lineOf(ev)}
            </li>
          ))
        )}
      </ul>

      {/* 로딩 오버레이: 레이아웃 영향 없음 */}
      {refreshing && (
        <div className="absolute inset-0 pointer-events-none flex items-start justify-end pr-1 pt-1">
          <div className="text-xs bg-black/60 text-white rounded px-2 py-1">Sync…</div>
        </div>
      )}

      {err && <p className="mt-2 text-sm text-red-500">{err}</p>}
    </div>
  );
}
