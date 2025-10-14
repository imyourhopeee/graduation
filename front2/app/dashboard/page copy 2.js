"use client";

import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import MjpegPlayer from "../components/MjpegPlayer";
import Navbar from "../components/Navbar";

const EVENT_AUTH = process.env.NEXT_PUBLIC_EVENT_AUTH ?? "none";
const AI_API = process.env.NEXT_PUBLIC_AI_URL ?? "http://localhost:3001"; // 스트림
const EVENT_API = process.env.NEXT_PUBLIC_EVENT_URL ?? "http://localhost:3002"; // 로그
const EVENT_PATH = process.env.NEXT_PUBLIC_EVENT_PATH ?? "/events";
const SOCKET_URL = process.env.NEXT_PUBLIC_SOCKET_URL || "http://localhost:3002";

// 침입 배너 유지 시간(ms)
const INTRUSION_HOLD_MS = Number(process.env.NEXT_PUBLIC_INTRUSION_HOLD_MS ?? 4000);

export default function DashboardPage() {
  const [lastStatus, setLastStatus] = useState(null); // 예: "200 OK"
  const [rawPreview, setRawPreview] = useState(""); // 응답 스니펫
  const [lastCount, setLastCount] = useState(null); // 파싱된 이벤트 개수
  const [isClient, setIsClient] = useState(false);
  const [eventLogs, setEventLogs] = useState([]); // ✅ 초기값 []
  const [err, setErr] = useState(null);
  const [hasToken, setHasToken] = useState(null); // ✅ 추가
  const pollTimerRef = useRef(null);

  // 배너 상태
  const [banner, setBanner] = useState(null); // { seatId, who, at } | null
  const bannerTimerRef = useRef(null);

  const streamSrc = `${AI_API.replace(/\/$/, "")}/stream/blur?cam=0&roi=1&intrusion=1&w=1280&h=960`;

  useEffect(() => setIsClient(true), []);

  // ✅ PATCH 1 (수정된 버전): 마운트 후 브라우저에서만 토큰 읽기
useEffect(() => {
  if (typeof window === "undefined") return;
  try {
    const storedToken =
      localStorage.getItem("access_token") ?? sessionStorage.getItem("access_token");
    setHasToken(Boolean(storedToken));
    if (!storedToken) setErr("No token found (localStorage/sessionStorage)");
  } catch {
    setHasToken(false);
  }
}, []);

  // --- PATCH 1 끝 ---

  useEffect(() => {
    if (!SOCKET_URL) {
      console.warn("SOCKET_URL is empty; skip connecting");
      return;
    }
    const socket = io(SOCKET_URL, {
      path: "/socket.io",
      transports: ["polling", "websocket"],
      withCredentials: true,
    });

    socket.on("connect", () => console.log("socket connected:", socket.id));
    socket.on("disconnect", () => console.log("socket disconnected"));

    // 서버가 실시간 이벤트를 푸시
    socket.on("event", (ev) => {
      setEventLogs((prev) => [lineOf(ev), ...prev].slice(0, 200));
      if (isIntrusion(ev)) showBannerFromEvent(ev);
    });
   

    socket.onAny((evt, ...args) => {
      console.log("[socket onAny]", evt, args?.[0]);
    });

    return () => socket.disconnect();
  }, []);

  // 공통: Unix(sec|ms|ISO) → epochSec 로 통일
  const toEpochSec = (t) => {
    if (t == null) return null;
    const n = Number(t);
    if (!Number.isNaN(n)) {
      // 초 단위인지 ms 단위인지 추정
      return n > 1e12 ? Math.floor(n / 1000) : n;
    }
    const ms = Date.parse(String(t));
    return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
  };

  const pick = (...vals) => vals.find(v => v !== undefined && v !== null && v !== "") ?? null;
  const getPerson = (ev) =>
    pick(ev.person_id, ev.identity, ev.user_label, ev.meta?.user_label);
  const getConf = (ev) =>
    pick(ev.confidence, ev.identity_conf, ev.meta?.confidence);
  const getSeat = (ev) =>
    pick(ev.seat_id, ev.zone_id, ev.meta?.seat_no, "-");
  const getCam = (ev) =>
    pick(ev.camera_id, ev.meta?.device_id, ev.meta?.camera_id, ev.cam, "-");
  const getWhenSec = (ev) =>
    pick(toEpochSec(ev.at), toEpochSec(ev.ended_at), toEpochSec(ev.started_at), Math.floor(Date.now()/1000));

  // 표시용 포맷터
  const fmt = (t) => {
    const s = toEpochSec(t);
    if (s == null) return "";
    return new Date(s * 1000).toLocaleString();
  };

const lineOf = (ev) => {
  const typ = String(ev.type ?? ev.event_type ?? "event").toLowerCase();
  // const seat = getSeat(ev);
  // const cam  = getCam(ev);
  const seat = ev.seat_id ?? ev.meta?.seat_no ?? "-";
  const cam  = ev.camera_id ?? ev.meta?.device_id ?? "-";

  //사람, 신뢰도
  // const who  = getPerson(ev);
  // const conf = getConf(ev);
  const who  = ev.person_id ?? ev.identity ?? ev.meta?.user_label ?? "Unknown";
  const conf = ev.confidence ?? ev.identity_conf ?? null;

  // const when = getWhenSec(ev);

  const whoTxt  = who ? ` · by ${who}` : "";
  // const confTxt = conf != null ? ` (${Number(conf).toFixed(2)})` : "";
  const confTxt = conf != null ? ` (${Math.round(conf * 100) / 100})` : "";

  // duration(초) 계산
  // const started = toEpochSec(ev.started_at);
  // const ended   = toEpochSec(ev.ended_at);
  // const durTxt  = (started && ended && ended >= started)
  //   ? ` · dur:${ended - started}s` : "";
  
  // if (typ === "intrusion") {
  // const whoTxt = who !== "Unknown" ? ` by ${who}` : "";
  // return `${fmt(when)} [Seat ${seat} · Cam ${cam}] Intrusion${whoTxt} (${conf ?? "-"}) · dur:${ev.duration_sec ?? "-"}s`;
  // }

  // 시간/지속시간
  const s = toEpochSec(ev.started_at);
  const e = toEpochSec(ev.ended_at);
  const when = e ?? toEpochSec(ev.at) ?? s ?? Math.floor(Date.now() / 1000);
  const dur =
    ev.duration_sec ??
    (s != null && e != null ? Math.max(0, e - s) : null);
  const durTxt = dur != null ? ` · dur:${dur}s` : "";

  if (typ === "intrusion") {
    return `${fmt(when)} [Seat ${seat} · Cam ${cam}] Intrusion by ${who}${confTxt}${durTxt}`;
  }
  if (typ === "intrusion_started") {
    return `${fmt(when)} [Seat ${seat} · Cam ${cam}] Intrusion STARTED`;
  }
  return `${fmt(when)} [${typ}] ${JSON.stringify(ev)}`;
};
//   // 그 외 이벤트 공통 포맷(디버깅용 필드도 함께) -> 오류 떠서 삭제함
//   return `${fmt(when)} [${typ}] seat=${seat} cam=${cam}${who ? ` person=${who}` : ""}`;
// };

 const isIntrusion = (ev) => {
   const t = String(ev?.type ?? ev?.event_type ?? "").toLowerCase();
   // 서버가 intrusion / intrusion_started / intrusion_triggered 등으로 보낼 수 있음
   return t === "intrusion" || t === "intrusion_started" || t === "intrusion_triggered";
 };


  const showBannerFromEvent = (ev) => {
    const seat = ev.zone_id ?? ev.seat_id ?? "-";
    const who = ev.user_label ?? ev.user ?? ev.identity ?? "Unknown";
    const atSec =
      toEpochSec(ev.ended_at) ?? toEpochSec(ev.at) ?? Math.floor(Date.now() / 1000);

    setBanner({ seatId: seat, who, at: atSec });

    if (bannerTimerRef.current) clearTimeout(bannerTimerRef.current);
    bannerTimerRef.current = setTimeout(() => setBanner(null), INTRUSION_HOLD_MS);
  };

  const maybeRaiseBannerFromList = (list) => {
    const arr = Array.isArray(list) ? [...list] : [];
    // ended_at / at 기준 최신 intrusion
    arr.sort((a, b) => {
      const ta = toEpochSec(a.ended_at) ?? toEpochSec(a.at) ?? 0;
      const tb = toEpochSec(b.ended_at) ?? toEpochSec(b.at) ?? 0;
      return tb - ta;
    });
    const latest = arr.find(isIntrusion);
    if (!latest) return;

    const nowMs = Date.now();
    const evMs =
      (toEpochSec(latest.ended_at) ??
        toEpochSec(latest.at) ??
        Math.floor(nowMs / 1000)) * 1000;

    if (nowMs - evMs <= INTRUSION_HOLD_MS) {
      showBannerFromEvent(latest);
    }
  };

  const fetchLogs = async () => {
    // 👉 진입 흔적 남기기 (패널에서 보이도록)
    setLastStatus("fetch-start");
    setRawPreview("");
    setLastCount(null);

    try {
      const url = `${EVENT_API.replace(/\/$/, "")}${EVENT_PATH}?limit=50`;

      // 토큰 읽기 (둘 다 확인)
      const token =
        typeof window !== "undefined"
          ? localStorage.getItem("access_token") ?? sessionStorage.getItem("access_token")
          : null;

      if (!token) {
        setErr("No token found (localStorage/sessionStorage)");
        setEventLogs([]);
        setLastStatus("no-token"); // ✅ 조기반환도 상태 남김
        setRawPreview("");
        setLastCount(0);
        return;
      }

      console.log("[dashboard] fetching", {
        url,
        tokenPrefix: token.slice(0, 12) + "...",
      });

      const res = await fetch(url, {
        method: "GET",
        headers: {
          Accept: "application/json",
          Authorization: `Bearer ${token}`,
        },
        credentials: "include",
      });

      const raw = await res.text();
      setLastStatus(`${res.status} ${res.statusText}`); // ✅ 응답 수신 흔적
      setRawPreview(raw.slice(0, 200));

      // HTML이면 인증/리다이렉트 응답일 수 있음
      if (raw.trim().startsWith("<")) {
        setErr("HTML received (possible auth redirect)");
        setEventLogs([]);
        setLastCount(0);
        return;
      }

      if (!res.ok) {
        setErr(`${res.status} ${res.statusText} - ${raw.slice(0, 300)}...`);
        setEventLogs([]);
        setLastCount(0);
        return;
      }

      // JSON → 배열 정규화
      let data;
      try {
        data = raw ? JSON.parse(raw) : [];
      } catch {
        data = null;
      }

      if (data == null) {
        const nd = [];
        for (const line of raw.split(/\r?\n/)) {
          const s = line.trim();
          if (!s) continue;
          try {
            nd.push(JSON.parse(s));
          } catch {}
        }
        data = nd.length ? nd : [];
      }

      const list = Array.isArray(data)
        ? data
        : Array.isArray(data?.items)
        ? data.items
        : Array.isArray(data?.data)
        ? data.data
        : Array.isArray(data?.rows)
        ? data.rows
        : Array.isArray(data?.events)
        ? data.events
        : [];

      if (!Array.isArray(list)) {
        const keys = data && typeof data === "object" ? Object.keys(data) : [];
        setErr(`Unexpected payload. keys=[${keys.join(", ")}]`);
        setEventLogs([]);
        setLastCount(0);
        return;
      }

      setLastCount(list.length); // ✅ 최종 카운트

      if (list.length === 0) {
        setErr(null);
        setEventLogs(["(no events yet)"]); // 데이터 없음 시에도 뭔가 보이게
        return;
      }

      maybeRaiseBannerFromList(list);

      const lines = list.slice().reverse().map(lineOf);
      setEventLogs(lines);
      setErr(null);
    } catch (e) {
      console.error("[dashboard] fetchLogs error:", e);
      setErr(e?.message || "failed");
      setEventLogs([]);
      setLastStatus("exception"); // ✅ 예외 흔적
      setRawPreview(String(e?.message || e).slice(0, 200));
      setLastCount(0);
    }
  };

  useEffect(() => {
    console.log("[dashboard] useEffect → fetchLogs()");
    fetchLogs(); // 즉시 1회
    pollTimerRef.current = setInterval(fetchLogs, 3000);

    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
      if (bannerTimerRef.current) clearTimeout(bannerTimerRef.current);
    };
    // 의존성 비우기: 마운트 1회만
  }, []);

  // ✅ 상태줄 문자열 (하이드레이션 안전)
  const statusLine = (() => {
    const url = `${EVENT_API.replace(/\/$/, "")}${EVENT_PATH}?limit=50`;
    const tokenTxt  = hasToken === null ? "…" : hasToken ? "yes" : "no";
    const statusTxt = lastStatus ?? "…";
    const countTxt  = lastCount ?? "…";
    return `url: ${url} · token:${tokenTxt} · status:${statusTxt} · count:${countTxt}`;
    // const rawTxt    = rawPreview ? rawPreview.slice(0, 200) : "";
    // return `url: ${url} · token:${tokenTxt} · status:${statusTxt} · count:${countTxt} · raw:"${rawTxt}"`;
  })();



  return (
    <main className="min-h-screen bg-gray-100 flex flex-col">
      <Navbar />

      {/* 침입 배너 */}
      {banner && (
        <div className="fixed top-16 left-0 right-0 z-50 px-4">
          <div className="mx-auto max-w-6xl rounded-xl shadow-lg border border-red-300 bg-red-600/90 text-white px-6 py-3">
            <div className="flex items-center justify-between">
              <div className="font-semibold text-lg">🚨 INTRUSION — Seat {banner.seatId}</div>
              <div className="text-sm opacity-90">{banner.who}</div>
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
        <div className="text-[11px] text-gray-500" suppressHydrationWarning>
          {statusLine}
        </div>
        <div className="w-full max-w-6xl bg-white rounded-xl shadow p-4 h-64 overflow-y-auto border border-gray-300">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold mb-2 text-gray-700">📋 이벤트 로그</h2>
            {err && <span className="text-sm text-red-600 mb-2">불러오기 오류: {err}</span>}
            <div className="text-[11px] text-gray-500">
              url: {`${EVENT_API.replace(/\/$/, "")}${EVENT_PATH}?limit=50`} · token:
              {typeof window !== "undefined" &&
              (localStorage.getItem("access_token") || sessionStorage.getItem("access_token"))
                ? "yes"
                : "no"}{" "}
              · status:{lastStatus ?? "-"} · count:{lastCount ?? "-"} · raw:"{rawPreview}"
            </div>
          </div>

          <ul className="space-y-2 text-sm text-white">
            {eventLogs.map((txt, i) => (
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
