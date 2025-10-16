// 대시보드 코드에서 그대로 가져온 버전 (동일 출력 보장)
export const toEpochSec = (t) => {
  if (t == null) return null;
  const n = Number(t);
  if (!Number.isNaN(n)) return n > 1e12 ? Math.floor(n / 1000) : n;
  const ms = Date.parse(String(t));
  return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
};

export const fmt = (t) => {
  const s = toEpochSec(t);
  if (s == null) return "";
  return new Date(s * 1000).toLocaleString();
};

export const tsOf = (ev) =>
  toEpochSec(ev.ended_at) ?? toEpochSec(ev.at) ?? toEpochSec(ev.started_at) ?? 0;

export const isIntrusion = (ev) => {
  const t = String(ev?.type ?? ev?.event_type ?? "").toLowerCase();
  return t === "intrusion" || t === "intrusion_started" || t === "intrusion_triggered";
};

export const lineOf = (ev) => {
  const typ = String(ev.type ?? ev.event_type ?? "event").toLowerCase();
  const seat = ev.seat_id ?? ev.meta?.seat_no ?? "-";
  const cam  = ev.camera_id ?? ev.meta?.device_id ?? "-";

  const who  = ev.person_id ?? ev.identity ?? ev.meta?.user_label ?? "Unknown";
  const conf = ev.confidence ?? ev.identity_conf ?? null;
  const confTxt = conf != null ? ` (${Math.round(conf * 100) / 100})` : "";

  const s = toEpochSec(ev.started_at);
  const e = toEpochSec(ev.ended_at);
  const when = e ?? toEpochSec(ev.at) ?? s ?? Math.floor(Date.now() / 1000);
  const dur = ev.duration_sec ?? (s != null && e != null ? Math.max(0, e - s) : null);
  const durTxt = dur != null ? ` · dur:${dur}s` : "";

  if (typ === "intrusion") {
    return `${fmt(when)} [Seat ${seat} · Cam ${cam}] Intrusion by ${who}${confTxt}${durTxt}`;
  }
  if (typ === "intrusion_started") {
    return `${fmt(when)} [Seat ${seat} · Cam ${cam}] Intrusion STARTED${confTxt}`;
  }
  return `${fmt(when)} [${typ}] ${JSON.stringify(ev)}`;
};
