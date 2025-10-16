"use client";

const EVENT_API  = process.env.NEXT_PUBLIC_EVENT_URL  ?? "http://localhost:3002";
const EVENT_PATH = process.env.NEXT_PUBLIC_EVENT_PATH ?? "/events";

function readRawToken() {
  try {
    return (
      localStorage.getItem("access_token") ||
      localStorage.getItem("token") ||
      localStorage.getItem("jwt") ||
      localStorage.getItem("authToken") ||
      sessionStorage.getItem("access_token") ||
      ""
    );
  } catch { return ""; }
}
const asBearer = (t) => (!t ? "" : t.startsWith("Bearer ") ? t : `Bearer ${t}`);

function withAuth(init = {}) {
  const bearer = asBearer(readRawToken());
  return {
    ...init,
    mode: "cors",
    headers: {
      ...(init.headers || {}),
      ...(bearer ? { Authorization: bearer } : {}),
      "Content-Type": "application/json",
    },
    cache: "no-store",
  };
}

// ---- robust array extractor ----
function asArray(data) {
  if (Array.isArray(data)) return data;
  if (data && typeof data === "object") {
    const keys = ["events","rows","result","data","items","list"];
    for (const k of keys) {
      if (Array.isArray(data[k])) return data[k];
    }
  }
  return [];
}

const toSec = (t) => {
  if (!t) return 0;
  const n = Number(t);
  if (!Number.isNaN(n)) return n > 1e12 ? Math.floor(n / 1000) : n;
  const ms = Date.parse(String(t));
  return Number.isNaN(ms) ? 0 : Math.floor(ms / 1000);
};
const tsOf = (ev) =>
  toSec(ev.ended_at) || toSec(ev.at) || toSec(ev.started_at) || toSec(ev.created_at);

export async function fetchEvents({ limit = 100 } = {}) {
  const url = `${EVENT_API}${EVENT_PATH}?limit=${limit}`;
  const res = await fetch(url, withAuth());
  if (!res.ok) {
    const e = new Error(`GET ${url} -> ${res.status}`);
    e.status = res.status;
    throw e;
  }

  // JSON 파싱 실패 대비: text → JSON 재시도
  let data;
  try { data = await res.json(); }
  catch {
    const txt = await res.text();
    try { data = JSON.parse(txt); } catch { data = []; }
  }

  const arr = asArray(data);
  // 최신이 위로
  arr.sort((a, b) => tsOf(b) - tsOf(a));
  return arr;
}

export function lineOf(ev) {
  const seat = ev.seat_id ?? ev.meta?.seat_no ?? "-";
  const cam  = ev.camera_id ?? ev.meta?.device_id ?? "-";
  const who  = ev.person_id ?? ev.identity ?? ev.meta?.user_label ?? "Unknown";
  const conf = ev.confidence ?? ev.identity_conf ?? null;
  const s = toSec(ev.started_at);
  const e = toSec(ev.ended_at);
  const when = e || toSec(ev.at) || s || Math.floor(Date.now()/1000);
  const dur = ev.duration_sec ?? (s && e ? Math.max(0, e - s) : null);

  const timeTxt = new Date(when * 1000).toLocaleString();
  const confTxt = conf != null ? ` (${Math.round(conf * 100) / 100})` : "";
  const durTxt  = dur != null ? ` - dur:${dur}s` : "";

  return `${timeTxt} [Seat ${seat} · Cam ${cam}] Intrusion by ${who}${confTxt}${durTxt}`;
}
