// controllers/eventController.js 
import { z } from "zod";
import { insertEvent, selectRecentIntrusions, selectEventById } from "../models/eventModel.js";
import { pool } from "../db/index.js";

export async function ingestDetections(req, res) {
  try {
    const payload = req.body;
    const count = Array.isArray(payload) ? payload.length : (payload ? 1 : 0);
    console.log("[/detections] received", count, "items");
    return res.status(200).json({ ok: true, count });
  } catch (e) {
    console.error("[/detections] error", e);
    return res.status(400).json({ ok: false, error: "bad_payload" });
  }
}
export async function listEvents(req, res) {
  try {
    const limit = Math.min(Number(req.query.limit || 50), 200);
    const q = `
      select
        id,
        lower(event_type) as event_type,
        started_at,
        ended_at,
        created_at,
        meta
      from events
      order by id desc
      limit $1
    `;
    const rs = await pool.query(q, [limit]);

    const items = rs.rows.map(r => {
      // at(초) 계산: ended > started > created
      const at =
        Math.floor(
          new Date(r.ended_at || r.started_at || r.created_at || Date.now()).getTime() / 1000
        );

      const m = typeof r.meta === "string" ? safeJson(r.meta) : (r.meta || {});
      return {
            id: r.id,
            type: r.event_type,           // 프론트가 type || event_type 사용
            event_type: r.event_type,
            seat_id: m.seat_no ?? null,   // ← meta에서 꺼냄
            camera_id: m.device_id ?? null,
            at,
            meta: m,
            started_at: r.started_at,
            ended_at: r.ended_at,
      };
    });

    return res.json(items);
  } catch (e) {
    console.error("[listEvents] error", e);
    return res.status(500).json({ ok: false, error: "query_failed" });
  }
}

// ✅ meta 문자열 방어 파서
function safeJson(x) {
  if (x == null) return {};
  if (typeof x !== "string") return x;
  try { return JSON.parse(x); } catch { return {}; }
}

// ✅ POST 본문 정규화 유틸 (현재 코드 유지)
function normalizeIncoming(b) {
  return {
    type: b.type || b.event_type,
    seat_id: b.seat_id ?? b?.meta?.seat_no ?? null,
    camera_id: b.camera_id ?? b?.meta?.device_id ?? null,
    at: b.at ?? Math.floor(Date.now() / 1000),
    meta: b.meta ?? {},
  };
}

// AI-server가 호출하는 이벤트 인서트 (침입 로그 포함)
const eventSchema = z.object({
  event_type: z.string(),                           // 'intrusion' 등
  started_at: z.string().or(z.date()).optional(),
  ended_at: z.string().or(z.date()).optional(),
  duration_sec: z.number().nullable().optional(),
  person_id: z.string().nullable().optional(),
  confidence: z.number().nullable().optional(),
  meta: z.object({
    seat_no: z.number(),                            // ★ 좌석 번호(필수)
    device_id: z.string().optional(),
    user_label: z.string().optional(),              // '홍길동' | 'unknown'
    phone_detected: z.boolean().optional(),
    thumb_path: z.string().optional(),
    bbox: z.any().optional()
  }).passthrough()
});

function toNumberOrUndef(v) {
  if (v === null || v === undefined || v === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}
function toBoolOrUndef(v) {
  if (v === null || v === undefined || v === "") return undefined;
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v !== 0;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (["1", "true", "yes", "y", "on"].includes(s)) return true;
    if (["0", "false", "no", "n", "off"].includes(s)) return false;
  }
  return undefined;
}
function parseAtSeconds(b) {
  const atNum = toNumberOrUndef(b.at);
  if (atNum !== undefined) return atNum;
  if (b.timestamp) {
    const ms = Date.parse(b.timestamp);
    if (!Number.isNaN(ms)) return Math.floor(ms / 1000);
  }
  const ended = toNumberOrUndef(b.ended_at);
  if (ended !== undefined) return ended;
  const started = toNumberOrUndef(b.started_at);
  if (started !== undefined) return started;
  return Math.floor(Date.now() / 1000);
}

export function normalizeEventBody(b = {}) {
  const out = { ...b };

  // 1) event_type/type 정규화
  const rawType = out.event_type ?? out.type ?? out.EventType ?? out.eventType;
  if (typeof rawType === "string") out.event_type = rawType.toLowerCase();
  else out.event_type = undefined;
  if (!out.type && out.event_type) out.type = out.event_type;

  // 2) seat_id 정규화
  const sid = Number(out.seat_id ?? out.zone_id ?? out.seatNo ?? out.seat ?? 0);
  out.seat_id = sid;

  // 3) camera_id 기본값
  if (!out.camera_id && typeof out.cameraId === "string") out.camera_id = out.cameraId;
  if (!out.camera_id) out.camera_id = "cam2";

  // 4) correlation_id 정규화
  if (!out.correlation_id && typeof out.correlationId === "string")
    out.correlation_id = out.correlationId;

  // 5) 시간 정규화
  const ms = Date.parse(out.timestamp ?? out.ended_at ?? out.started_at ?? "");
  out.at = !Number.isNaN(ms) ? Math.floor(ms / 1000) : Math.floor(Date.now() / 1000);

  // 6) 필드명 매핑
  if (out.identity && !out.person_id) out.person_id = out.identity;
  if (out.identity_conf != null && out.confidence == null)
    out.confidence = Number(out.identity_conf);
  const phoneDetected =
    typeof out.phone_capture === "boolean" ? out.phone_capture : out.phone_capture === "true";
  if (out.meta == null || typeof out.meta !== "object") out.meta = {};
  if (phoneDetected) out.meta.phone_detected = phoneDetected;

  // 7) meta 필수 보완
  out.meta.seat_no = sid;
  if (!out.meta.device_id && out.camera_id) out.meta.device_id = out.camera_id;
  if (!out.meta.user_label && out.identity) out.meta.user_label = out.identity;

  // 8) 정리
  delete out.identity;
  delete out.identity_conf;
  delete out.phone_capture;
  delete out.zone_id;
  delete out.timestamp;

  return out;
}

export async function addEvent(req, res) {
  const normalized = normalizeEventBody(req.body || {});
  console.log("[/events] body(raw) =", JSON.stringify(req.body));
  console.log("[/events] body(norm) =", JSON.stringify(normalized));

  if (!normalized.type && normalized.event_type) normalized.type = normalized.event_type;

  const nowSec = Math.floor(Date.now() / 1000);
  const atSec  = Number.isFinite(normalized.at) ? normalized.at : nowSec;
  const atISO  = new Date(atSec * 1000).toISOString();

  if (!normalized.started_at && !normalized.ended_at) {
    if (normalized.type?.endsWith("_started")) {
      normalized.started_at = atISO;
    } else if (normalized.type?.endsWith("_ended")) {
      normalized.ended_at = atISO;
    } else {
      normalized.started_at = atISO;
    }
  }

  const parsed = eventSchema.safeParse(normalized);
  if (!parsed.success) {
    try { console.warn("[/events] zod issues:", JSON.stringify(parsed.error.issues)); } catch {}
    return res.status(400).json({ ok: false, error: "invalid_body" });
  }

  try {
    const data = parsed.data;

    if (!data.type && data.event_type) data.type = data.event_type;
    if (!data.started_at && !data.ended_at && typeof normalized.at === 'number') {
      data.ended_at = new Date(normalized.at * 1000).toISOString();
    }
    if (!data.duration_sec && data.started_at && data.ended_at) {
      const s = new Date(data.started_at).getTime();
      const e = new Date(data.ended_at).getTime();
      if (Number.isFinite(s) && Number.isFinite(e) && e >= s) {
        data.duration_sec = Math.round((e - s) / 1000);
      }
    }

    // DB 저장
    const id = await insertEvent(data);
    console.log("[addEvent] inserted id=%s type=%s seat=%s cam=%s",
      id, data.type, data.seat_id, data.camera_id);

    // 방금 저장한 행 재조회 (db.query → 모델 함수 사용)
    const r = await selectEventById(id);
    if (!r) {
      return res.status(201).json({ ok: true, id });
    }

    // 실시간 소켓 브로드캐스트
    // try {
    //   const io = req.app.get("socket");
    //   if (io) {
    //     io.emit("event", {
    //       type: r.type,
    //       seat_id: r.seat_id ?? null,
    //       camera_id: r.camera_id ?? null,
    //       at: Math.floor(
    //         new Date(r.created_at || r.started_at || r.ended_at || atISO).getTime() / 1000
    //       ),
    //       meta: typeof r.meta === "string" ? safeJson(r.meta) : (r.meta ?? {}),
    //       started_at: r.started_at,
    //       ended_at: r.ended_at,
    //     });
    //   }
    // } catch (e) {
    //   console.warn("[addEvent] socket emit skipped:", e?.message);
    // }
    try {
      const io = req.app.get("socket"); // app.js에서 app.set("socket", socketApi)
      if (!io) throw new Error("socket instance not found");

      // 원본 r 보강해서 프론트가 쓰기 쉬운 공통 형태로 정규화
      const atSec =
        Math.floor(
          new Date(
            r.at ||
            r.created_at ||
            r.ended_at ||
            r.started_at ||
            atISO // 너희 코드 상단에서 만든 ISO
          ).getTime() / 1000
        ) || Math.floor(Date.now() / 1000);

      const ev = {
        id: r.id ?? undefined,
        type: (r.type ?? r.event_type ?? "").toLowerCase(),   // "intrusion_started" 등
        event_type: (r.type ?? r.event_type ?? "").toLowerCase(),
        seat_id: r.seat_id ?? r.meta?.seat_no ?? null,
        camera_id: r.camera_id ?? r.meta?.device_id ?? null,
        correlation_id: r.correlation_id ?? r.meta?.correlation_id ?? undefined,
        at: atSec,
        meta: typeof r.meta === "string" ? safeJson(r.meta) : (r.meta ?? {}),
        started_at: r.started_at ?? null,
        ended_at: r.ended_at ?? null,
      };
      io.emit("event", ev);

      // 2) 타입별 채널도 함께 (프론트가 특정 채널만 듣는 경우 대비)
      if (ev.type) io.emit(ev.type, ev);
      if (ev.event_type && ev.event_type !== ev.type) io.emit(ev.event_type, ev);
    } catch (e) {
      console.warn("[addEvent] socket emit skipped:", e?.message);
    }


    // 최종 응답
    return res.status(201).json({
      ok: true,
      id: r.id,
      type: r.type,
      seat_id: r.seat_id ?? null,
      camera_id: r.camera_id ?? null,
      at: Math.floor(
        new Date(r.created_at || r.started_at || r.ended_at || atISO).getTime() / 1000
      ),
      meta: typeof r.meta === "string" ? safeJson(r.meta) : (r.meta ?? {}),
      started_at: r.started_at,
      ended_at: r.ended_at,
    });

  } catch (e) {
    console.error("[addEvent] failed:", e);
    return res.status(500).json({ ok: false, error: "insert_failed" });
  }
}

// 프론트 대시보드: 최근 침입 전용 조회
export async function listIntrusions(req, res) {
  let sinceISO;
  const q = String(req.query.since || "24h");
  if (/\d+h$/i.test(q)) {
    const h = parseInt(q, 10);
    sinceISO = new Date(Date.now() - h * 3600 * 1000).toISOString();
  } else {
    sinceISO = new Date(q).toISOString();
  }

  try {
    const rows = await selectRecentIntrusions(sinceISO);
    return res.json({ ok: true, items: rows });
  } catch (e) {
    console.error("[listIntrusions] failed:", e);
    return res.status(500).json({ ok: false, error: "query_failed" });
  }
}
