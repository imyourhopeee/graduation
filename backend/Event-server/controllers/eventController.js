// controllers/eventController.js
import { z } from "zod";
import { insertEvent, selectRecentIntrusions, selectEventById } from "../models/eventModel.js";
import { query } from "../db/index.js"; // ✅ pool 대신 query 사용

// ---- 유틸 ----
function safeJson(x) {
  if (x == null) return {};
  if (typeof x !== "string") return x;
  try { return JSON.parse(x); } catch { return {}; }
}

// ---- AI → 덤프 수신(바디만 확인) ----
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

// ---- 이벤트 목록 (가벼운 목록: meta 제외) ----
// ※ 침입만 보려면 아래 listIntrusions 사용
export async function listEvents(req, res) {
  try {
    const rawLimit = Number.parseInt(req.query.limit, 10);
    const limit = Math.max(1, Math.min(Number.isFinite(rawLimit) ? rawLimit : 50, 200));
    const rawCursor = Number.parseInt(req.query.cursor, 10);
    const cursor = Number.isFinite(rawCursor) ? rawCursor : null;

    const params = [];
    let sql = `
      SELECT
        id,
        event_type AS type,
        meta->>'seat_no'   AS seat_id,
        meta->>'device_id' AS camera_id,
        person_id,
        confidence,
        duration_sec,
        started_at,
        ended_at,
        COALESCE(ended_at, started_at, created_at) AS at
      FROM events
    `;
    //위 목록: 원래는 created_at AS at 였는데 수정함

    if (cursor) { params.push(cursor); sql += ` WHERE id < $${params.length}`; }
    params.push(limit);
    sql += ` ORDER BY id DESC LIMIT $${params.length}`;

    const rs = await query(sql, params);
    return res.json({ events: rs.rows, nextCursor: rs.rows.length ? rs.rows[rs.rows.length - 1].id : null });
  } catch (e) {
    console.error("[listEvents] error", e);
    return res.status(500).json({ ok: false, error: "query_failed" });
  }
}

// ---- POST 바디 정규화 → DB 스키마에 맞춤 ----
const eventSchema = z.object({
  event_type: z.string(), // 'intrusion' 등
  started_at: z.string().or(z.date()).optional(),
  ended_at: z.string().or(z.date()).optional(),
  duration_sec: z.number().nullable().optional(),
  person_id: z.string().nullable().optional(),
  confidence: z.number().nullable().optional(),
  meta: z.object({
    seat_no: z.number(),               // ← DB에는 seat_id 컬럼이 없으므로 meta.seat_no로 보관
    device_id: z.string().optional(),
    user_label: z.string().optional(),
    phone_detected: z.boolean().optional(),
    thumb_path: z.string().optional(),
    bbox: z.any().optional(),
  }).passthrough(),
});

export function normalizeEventBody(b = {}) {
  const out = { ...b };

  // 1) event_type/type 정규화 (소문자)
  const rawType = out.event_type ?? out.type ?? out.EventType ?? out.eventType;
  out.event_type = typeof rawType === "string" ? rawType.toLowerCase() : undefined;
  if (!out.type && out.event_type) out.type = out.event_type;

  // 2) 좌석/카메라를 meta로 수렴 (DB에 전용 컬럼 없음)
  const sid = Number(out.seat_id ?? out.zone_id ?? out.seatNo ?? out.seat ?? 0);
  if (!out.meta || typeof out.meta !== "object") out.meta = {};
  out.meta.seat_no = Number.isFinite(sid) ? sid : 0;
  if (!out.meta.device_id && out.camera_id) out.meta.device_id = String(out.camera_id);
  if (!out.meta.user_label && out.identity) out.meta.user_label = String(out.identity);

  // 3) 사람/신뢰도 보정
  if (out.identity && !out.person_id) out.person_id = out.identity;
  if (out.identity_conf != null && out.confidence == null) out.confidence = Number(out.identity_conf);

  // 4) 시간 보정: created_at은 DB default, 필요 시 started/ended만 설정
  const ms = Date.parse(out.timestamp ?? out.ended_at ?? out.started_at ?? "");
  out.at = !Number.isNaN(ms) ? Math.floor(ms / 1000) : Math.floor(Date.now() / 1000);

  // 5) 정리
  delete out.identity;
  delete out.identity_conf;
  delete out.phone_capture;
  delete out.zone_id;
  delete out.timestamp;

  return out;
}

// ---- 이벤트 생성 ----
export async function addEvent(req, res) {
  const normalized = normalizeEventBody(req.body || {});
  console.log("[/events] body(raw) =", JSON.stringify(req.body));
  console.log("[/events] body(norm) =", JSON.stringify(normalized));

  // started/ended 기본값 보정
  const nowSec = Math.floor(Date.now() / 1000);
  const atSec  = Number.isFinite(normalized.at) ? normalized.at : nowSec;
  const atISO  = new Date(atSec * 1000).toISOString();

  if (!normalized.started_at && !normalized.ended_at) {
    if (normalized.type?.endsWith("_started") || normalized.event_type?.endsWith("_started")) {
      normalized.started_at = atISO;
    } else if (normalized.type?.endsWith("_ended") || normalized.event_type?.endsWith("_ended")) {
      normalized.ended_at = atISO;
    } else {
      normalized.started_at = atISO;
    }
  }

  // 추가 - zod 이전 사전 보정 추가 시작
  const toISO = (v) => {
    if (!v) return null;
    if (typeof v === "number") {
      const ms = v > 1e12 ? v : v * 1000;
      return new Date(ms).toISOString();
    }
    const ms = Date.parse(v);
    return Number.isNaN(ms) ? null : new Date(ms).toISOString();
  };

  // TIMESTAMPTZ 컬럼과 호환되도록 ISO 문자열로 통일
  if (normalized.started_at) normalized.started_at = toISO(normalized.started_at);
  if (normalized.ended_at)   normalized.ended_at   = toISO(normalized.ended_at);
  
  // duration_sec → 정수 강제
    if (normalized.duration_sec != null) {
      const n = Number(normalized.duration_sec);
      normalized.duration_sec = Number.isFinite(n) ? Math.round(n) : null;
    } //여기까지 추가한것.

  // zod 검증 (DB 스키마에 맞는 필드만)
  const parsed = eventSchema.safeParse(normalized);
  if (!parsed.success) {
    try { console.warn("[/events] zod issues:", JSON.stringify(parsed.error.issues)); } catch {}
    return res.status(400).json({ ok: false, error: "invalid_body" });
  }

  try {
    const data = parsed.data;

    // (기존) duration 자동 계산
    if (!data.duration_sec && data.started_at && data.ended_at) {
      const s = new Date(data.started_at).getTime();
      const e = new Date(data.ended_at).getTime();
      if (Number.isFinite(s) && Number.isFinite(e) && e >= s) {
        data.duration_sec = Math.round((e - s) / 1000);
      }
    }

    // 추가: 혹시라도 float/문자열이 끼어들었을 때를 위한 이중 안전망
    if (data.duration_sec != null) {
      const n = Number(data.duration_sec);
      data.duration_sec = Number.isFinite(n) ? Math.round(n) : null;
    }
    // Date 객체가 들어오는 경우를 대비해 ISO로 고정
    const asISO = (v) => (v instanceof Date ? v.toISOString() : v);
    if (data.started_at) data.started_at = asISO(data.started_at);
    if (data.ended_at)   data.ended_at   = asISO(data.ended_at);
    // 추가 끝 ===

    // DB 저장 (models/eventModel.js 의 insertEvent는 DB 컬럼에 맞춰야 함)
    const id = await insertEvent(data);
    console.log("[addEvent] inserted id=%s event_type=%s seat=%s cam=%s",
      id, data.event_type, data.meta?.seat_no, data.meta?.device_id);

    // 저장된 행 재조회(모델 함수 사용) 
    // — 컬럼/별칭은 모델에서 맞춰도 되고 여기서 후처리해도 됨
    const r = await selectEventById(id); // r에는 event_type, created_at, meta 등이 있어야 함
    if (!r) return res.status(201).json({ ok: true, id });

    // 소켓 브로드캐스트 (정규화)
    try {
      const io = req.app.get("io");
      if (io) {
        // at 계산 보강: 숫자면 그대로(초 단위), 문자열/Date면 파싱
        const atRaw =
          r.at ??
          r.created_at ??
          r.ended_at ??
          r.started_at ??
          Date.now(); // atISO 미정의 방지

        const atEpoch =
          typeof atRaw === "number"
            ? atRaw
            : Math.floor(new Date(atRaw).getTime() / 1000);

        const ev = {
          id: r.id,
          type: String(r.type ?? r.event_type ?? "").toLowerCase(),
          event_type: String(r.type ?? r.event_type ?? "").toLowerCase(),
          seat_id: r.seat_id ?? r.meta?.seat_no ?? null,
          camera_id: r.camera_id ?? r.meta?.device_id ?? null,
          person_id: r.person_id ?? null,
          confidence: r.confidence ?? null,
          duration_sec: r.duration_sec ?? null,
          started_at: r.started_at ?? null,
          ended_at: r.ended_at ?? null,
          at: atEpoch,
          meta:
            typeof r.meta === "string" ? safeJson(r.meta) : (r.meta ?? {}),
        };

        // ✅ 채널명 통일 (대시보드 룸으로)
        io.to("dashboard").emit("events:new", ev);
        if (ev.type) io.to("dashboard").emit(`events:${ev.type}`, ev);
      }
    } catch (e) {
      console.warn("[addEvent] socket emit skipped:", e?.message);
    }


    // 최종 응답: 목록과 동일한 키로 반환
    return res.status(201).json({
      ok: true,
      id: r.id,
      type: (r.type ?? r.event_type ?? "").toLowerCase(),
      seat_id: r.seat_id ?? r.meta?.seat_no ?? null,
      camera_id: r.camera_id ?? r.meta?.device_id ?? null,
      person_id: r.person_id ?? null,
      confidence: r.confidence ?? null,
      duration_sec: r.duration_sec ?? null,
      started_at: r.started_at ?? null,
      ended_at: r.ended_at ?? null,
      at: Math.floor(
        new Date(r.at || r.created_at || r.ended_at || r.started_at || atISO).getTime() / 1000
      ),
      meta: typeof r.meta === "string" ? safeJson(r.meta) : (r.meta ?? {}),
    });

  } catch (e) {
    console.error("[addEvent] failed:", e);
    return res.status(500).json({ ok: false, error: "insert_failed" });
  }
}

// ---- 최근 침입 전용(필요 시 유지) ----
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
    const rows = await selectRecentIntrusions(sinceISO); // 모델에서 event_type/created_at 기준으로 구현되어야 함
    return res.json({ ok: true, items: rows });
  } catch (e) {
    console.error("[listIntrusions] failed:", e);
    return res.status(500).json({ ok: false, error: "query_failed" });
  }
}
