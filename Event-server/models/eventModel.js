// models/eventModel.js
import { pool } from "../db/index.js";

// ──────────────────────────────────────────────────────────────
// 현재 events 테이블 컬럼(스크린샷 기준):
// id, started_at, ended_at, duration_sec, meta(JSONB), created_at,
// event_type, person_id, confidence
// ──────────────────────────────────────────────────────────────

// ✅ DB에 들어갈 행으로 컨트롤러 payload를 정규화 (핵심)
function mapPayloadToRow(evt = {}) {
  // controller.normalizeEventBody() 기준 매핑
  const event_type =
    typeof evt.event_type === "string"
      ? evt.event_type
      : typeof evt.type === "string"
      ? evt.type
      : undefined;

  // meta JSONB 구성
  const seat_no =
    evt.seat_id ?? evt?.meta?.seat_no ?? null;
  const device_id =
    evt.camera_id ?? evt?.meta?.device_id ?? null;

  const meta = {
    ...(evt.meta || {}),
    ...(seat_no != null ? { seat_no: Number(seat_no) } : {}),
    ...(device_id != null ? { device_id: String(device_id) } : {}),
  };

  const row = {
    event_type,                                  // TEXT
    started_at: evt.started_at ?? null,          // TIMESTAMPTZ
    ended_at: evt.ended_at ?? null,              // TIMESTAMPTZ
    duration_sec: evt.duration_sec ?? null,      // INT
    person_id: evt.person_id ?? null,            // TEXT
    confidence: evt.confidence ?? null,          // FLOAT
    meta,                                        // JSONB
    // created_at: DB default NOW()
  };

  // controller에서 at만 있고 started/ended 없을 때 ended_at 채우는 로직이 있으므로
  // 여긴 그대로 둡니다(컨트롤러에서 이미 보정했다는 가정).
  return row;
}

// 실제 DB에 존재하는 컬럼만 명시
const EVENT_COLUMNS = [
  "event_type",
  "started_at",
  "ended_at",
  "duration_sec",
  "person_id",
  "confidence",
  "meta",
  // created_at 은 DB DEFAULT NOW()면 생략
];

/** 안전한 INSERT: DB 컬럼 화이트리스트만 사용 */
export async function saveEvent(row, client = pool) {
  const cols = [];
  const vals = [];
  const phs = [];

  for (const col of EVENT_COLUMNS) {
    if (row[col] !== undefined) {
      cols.push(col);
      vals.push(row[col]);
      phs.push(`$${vals.length}`);
    }
  }
  if (cols.length === 0) throw new Error("No valid event fields to insert");

  const sql = `
    INSERT INTO events (${cols.join(",")})
    VALUES (${phs.join(",")})
    RETURNING id
  `;
  const { rows } = await client.query(sql, vals);
  return rows[0].id;
}

/** 컨트롤러에서 사용하는 단건 INSERT */
export async function insertEvent(evt) {
  const row = mapPayloadToRow(evt);     // ✅ 컨트롤러 payload → DB행
  return saveEvent(row);
}

export async function saveEventsBulk(events) {
  if (!events?.length) return [];
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const ids = [];
    for (const e of events) ids.push(await insertEvent(e)); // ✅ mapPayloadToRow 경유
    await client.query("COMMIT");
    return ids;
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  } finally {
    client.release();
  }
}

/** id로 조회 (컨트롤러가 원하는 형태로 가공해서 반환) */
export async function selectEventById(id) {
  const q = `
    SELECT id, event_type, started_at, ended_at, duration_sec,
           person_id, confidence, meta, created_at
      FROM events
     WHERE id = $1
  `;
  const { rows } = await pool.query(q, [id]);
  const r = rows[0];
  if (!r) return null;

  // 컨트롤러가 기대하는 필드로 재가공
  const type = r.event_type;
  const seat_id =
    r.meta && r.meta.seat_no != null ? Number(r.meta.seat_no) : null;
  const camera_id =
    r.meta && r.meta.device_id != null ? String(r.meta.device_id) : null;

  return {
    id: r.id,
    type,
    seat_id,
    camera_id,
    meta: r.meta || {},
    started_at: r.started_at,
    ended_at: r.ended_at,
    duration_sec: r.duration_sec,
    person_id: r.person_id,
    confidence: r.confidence,
    created_at: r.created_at,
  };
}

/** 최근 침입 조회 (프론트 대시보드용) */
export async function selectRecentIntrusions(sinceISO) {
  const { rows } = await pool.query(
    `SELECT e.id, e.event_type, e.started_at, e.ended_at, e.duration_sec,
            e.person_id, e.confidence, e.meta,
            s.seat_no, s.owner_user_id, s.name AS seat_name
       FROM events e
       LEFT JOIN seats s
         ON s.seat_no = (e.meta->>'seat_no')::int
      WHERE e.event_type = 'intrusion'
        AND e.started_at >= $1
      ORDER BY e.started_at DESC`,
    [sinceISO]
  );
  return rows;
}
