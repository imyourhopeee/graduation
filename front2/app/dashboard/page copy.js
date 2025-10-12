"use client";

import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import MjpegPlayer from "../components/MjpegPlayer";
import Navbar from "../components/Navbar";

const AI_API = process.env.NEXT_PUBLIC_AI_URL ?? "http://localhost:3001";       // 스트림
const EVENT_API = process.env.NEXT_PUBLIC_EVENT_URL ?? "http://localhost:3002"; // 로그
const EVENT_PATH = process.env.NEXT_PUBLIC_EVENT_PATH ?? "/events";
const SOCKET_URL = process.env.NEXT_PUBLIC_SOCKET_URL|| "http://localhost:3002";

// NEW: 침입 배너 유지 시간(ms)
const INTRUSION_HOLD_MS = Number(process.env.NEXT_PUBLIC_INTRUSION_HOLD_MS ?? 4000);

export default function DashboardPage() {
  const [isClient, setIsClient] = useState(false);
  const [eventLogs, setEventLogs] = useState(null); // string[] or null
  const [err, setErr] = useState(null);
  const timerRef = useRef(null);

 // NEW: 배너 상태
  const [banner, setBanner] = useState(null); // { seatId, who, at } | null
  const bannerTimerRef = useRef(null);

  const streamSrc = `${AI_API.replace(/\/$/, "")}/stream/blur?cam=0&roi=1&w=1280&h=960`;

  useEffect(() => setIsClient(true), []);
  useEffect(() => {
    const url = SOCKET_URL;
    if (!url) {
      console.warn("SOCKET_URL is empty; skip connecting");
      return;
    }
    console.log("connecting socket to:", url);
    const socket = io(url, {
      transports: ["polling", "websocket"],
      withCredentials: true,
      upgrade: true,  
      path: "/socket.io",
    });
    socket.on("connect", () => console.log("socket connected:", socket.id));
    socket.on("disconnect", () => console.log("socket disconnected"));
    return () => socket.disconnect();
  }, []);

  // Unix(sec|ms) → 로컬 문자열
  const fmt = (t) => {
    if (t == null) return "";
    let n = Number(t);
    if (Number.isNaN(n)) return String(t);
    if (n > 1e12) n = Math.floor(n / 1000); // ms → sec
    const d = new Date(n * 1000);
    return d.toLocaleString();
  };

  // 이벤트 한 줄로 변환
  const lineOf = (ev) => {
    const typ = ev.type ?? ev.event_type ?? "event";
    if (typ === "intrusion") {
      const seat = ev.zone_id ?? ev.seat_id ?? "-";
      const who = ev.user_label ?? ev.user ?? "Unknown";
      const s = Number(ev.started_at || 0);
      const e = Number(ev.ended_at || 0);
      const dwell = Math.max(0, Math.round(e - s));
      return `${fmt(e || Date.now()/1000)}  [Seat ${seat}] Intrusion by ${who} (dwell ${dwell}s)`;
    }
    return `${fmt(ev.ended_at ?? ev.at ?? Date.now()/1000)}  [${typ}] ${JSON.stringify(ev)}`;
  };

  // NEW: intrusion 판단 헬퍼
  const isIntrusion = (ev) => {
    const typ = ev?.type ?? ev?.event_type;
    return String(typ).toLowerCase() === "intrusion";
  };

  // NEW: 이벤트 → 배너 표시
  const showBannerFromEvent = (ev) => {
    const seat = ev.zone_id ?? ev.seat_id ?? "-";
    const who = ev.user_label ?? ev.user ?? "Unknown";
    const atSec = Number(ev.ended_at ?? ev.at ?? Date.now() / 1000);
    setBanner({ seatId: seat, who, at: atSec });

    if (bannerTimerRef.current) clearTimeout(bannerTimerRef.current);
    bannerTimerRef.current = setTimeout(() => setBanner(null), INTRUSION_HOLD_MS);
  };

  // NEW: 가장 최신 intrusion 이벤트를 찾아 배너 띄우기
  const maybeRaiseBannerFromList = (list) => {
    // 최신 순으로 정렬된 배열 가정이 없으면 정렬
    const arr = Array.isArray(list) ? [...list] : [];
    // ended_at or at 기준 가장 최근 intrusion 찾기
    arr.sort((a, b) => {
      const ta = Number(a.ended_at ?? a.at ?? 0);
      const tb = Number(b.ended_at ?? b.at ?? 0);
      return tb - ta;
    });
    const latest = arr.find(isIntrusion);
    if (!latest) return;

    const nowMs = Date.now();
    const evMs = (Number(latest.ended_at ?? latest.at ?? nowMs / 1000)) * 1000;

    // 너무 예전 이벤트면 배너 생략(최근 INTRUSION_HOLD_MS 이내만 보여줌)
    if (nowMs - evMs <= INTRUSION_HOLD_MS) {
      showBannerFromEvent(latest);
    }
  };

  const fetchLogs = async () => {
    try {
      const url = `${EVENT_API.replace(/\/$/, "")}${EVENT_PATH}?limit=50`;
      const token =
        typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
      const headers = token ? { Authorization: `Bearer ${token}` } : {};
      const res = await fetch(url, {
        method: "GET",
        headers: {
          Accept: "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
          // credentials: "include",
        },
      });
      console.log("[dashboard] GET /events status =", res.status);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const data = await res.json();
      const list = Array.isArray(data) ? data : (Array.isArray(data?.items) ? data.items : []);
      const lines = list.slice().reverse().map(lineOf);
      setEventLogs(lines);
      setErr(null);

      // NEW: 로그 목록으로부터 배너 갱신 시도
      maybeRaiseBannerFromList(list);

    } catch (e) {
      setErr(e?.message || "failed");
      if (eventLogs == null) {
        setEventLogs([
          "예시) 시스템 시작됨",
          "예시) Seat 0에서 타인 30초 머무름 → 침입",
        ]);
      }
    }
  };

  useEffect(() => {
    fetchLogs(); // 즉시 1회
    timerRef.current = setInterval(fetchLogs, 3000); // 3초마다 폴링
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="min-h-screen bg-gray-100 flex flex-col">
      <Navbar />

      {/* NEW: 침입 배너 */}
      {banner && (
        <div className="fixed top-16 left-0 right-0 z-50 px-4">
          <div className="mx-auto max-w-6xl rounded-xl shadow-lg border border-red-300 bg-red-600/90 text-white px-6 py-3">
            <div className="flex items-center justify-between">
              <div className="font-semibold text-lg">
                🚨 INTRUSION — Seat {banner.seatId}
              </div>
              <div className="text-sm opacity-90">
                {banner.who}
              </div>
            </div>
          </div>
        </div>
      )}

      <section className="flex-grow flex flex-col items-center justify-start p-10 gap-6">
        {/* 실시간 카메라 카드 */}
        <div className="w-full max-w-6xl bg-white rounded-xl shadow-xl p-6">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-3xl font-bold text-gray-800">실시간 카메라 영상</h1>
              </div>

          {isClient && (
            <div className="flex justify-center">
              <MjpegPlayer src={streamSrc} style={{ height: "70vh" }} />
            </div>
          )}
        </div>

        {/* 이벤트 로그 카드 */}
        <div className="w-full max-w-6xl bg-white rounded-xl shadow p-4 h-64 overflow-y-auto border border-gray-300">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold mb-2 text-gray-700">📋 이벤트 로그</h2>
            {err && <span className="text-sm text-red-600 mb-2">불러오기 오류: {err}</span>}
          </div>
          <ul className="space-y-2 text-sm text-white">
            {(eventLogs ?? []).map((txt, i) => (
              <li key={i} className="bg-black px-4 py-2 rounded-lg shadow-sm border border-gray-700">
                {txt}
              </li>
            ))}
          </ul>
        </div>
      </section>
    </main>
  );
}
