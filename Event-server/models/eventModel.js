// models/eventModel.js
import { pool } from "../db/index.js";

// ──────────────────────────────────────────────────────────────
// events 테이블 컬럼(현재 스키마):
// id, started_at, ended_at, duration_sec, meta(JSONB), created_at,
// event_type, person_id, confidence
// ──────────────────────────────────────────────────────────────
const JSONB_COLS = new Set(["person_ids", "confidences", "meta"]);

// 숫자/문자/Date를 TIMESTAMPTZ ISO로 정규화
function toIsoOrNull(v) {
  if (v == null || v === "") return null;
  if (v instanceof Date) return v.toISOString();
  const n = Number(v);
  if (Number.isFinite(n)) {
    // 초 단위/밀리초 단위 추정
    const ms = n > 1e12 ? n : n * 1000;
    return new Date(ms).toISOString();
  }
  const ms = Date.parse(String(v));
  return Number.isNaN(ms) ? null : new Date(ms).toISOString();
}

// ✅ 컨트롤러 payload → DB 행 매핑 (스키마에 맞춤)
function mapPayloadToRow(evt = {}) {
  const rawType =
    typeof evt.event_type === "string"
      ? evt.event_type
      : typeof evt.type === "string"
      ? evt.type
      : undefined;

  // event_type은 소문자로 통일 저장
  const event_type = rawType ? String(rawType).toLowerCase() : undefined;

  // meta JSONB 구성 (seat_no/device_id를 meta로 수렴)
  const seat_no = evt.seat_id ?? evt?.meta?.seat_no ?? null;
  const device_id = evt.camera_id ?? evt?.meta?.device_id ?? null;

  const meta = {
    ...(evt.meta || {}),
    ...(seat_no != null ? { seat_no: Number(seat_no) } : {}),
    ...(device_id != null ? { device_id: String(device_id) } : {}),
  };
   // 배열 보정
   const toArr = (v) => Array.isArray(v) ? v : (v == null ? [] : [v]);
   const person_ids  = toArr(evt.person_ids ?? evt.person_id);
   const confidences = toArr(evt.confidences ?? evt.confidence);
 
   // 표시용 user_label 자동 채움(없으면)
   if (!meta.user_label && person_ids.length) {
     meta.user_label = person_ids.join(", ");
   }
   
  return {
    event_type,                                  // TEXT
    started_at: toIsoOrNull(evt.started_at),     // TIMESTAMPTZ
    ended_at: toIsoOrNull(evt.ended_at),         // TIMESTAMPTZ
    duration_sec:
      evt.duration_sec == null ? null : Number(evt.duration_sec), // INT
    person_ids,  
    confidences,
    meta,                              // JSONB
    // created_at: DB default NOW()
  };
}

// 실제 DB에 존재하는 컬럼만 명시
const EVENT_COLUMNS = [
  "event_type",
  "started_at",
  "ended_at",
  "duration_sec",
  "person_ids",
  "confidences",
  "meta",
];

/** 안전 INSERT: 허용된 컬럼만 저장 */
export async function saveEvent(row, client = pool) {
  const cols = [];
  const vals = [];
  const phs = [];

  for (const col of EVENT_COLUMNS) {
    if (row[col] !== undefined) {
      cols.push(col);
      if (JSONB_COLS.has(col)) {
       // jsonb 컬럼: 문자열로 직렬화 + ::jsonb 캐스팅
        vals.push(JSON.stringify(row[col] ?? null));
        phs.push(`$${vals.length}::jsonb`);
      } else {
        vals.push(row[col]);
        phs.push(`$${vals.length}`);
      }      
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

/** 컨트롤러에서 사용하는 단건 INSERT (트랜잭션용 client 전달 가능) */
export async function insertEvent(evt, client = pool) {
  const row = mapPayloadToRow(evt);
  return saveEvent(row, client);
}

/** 대량 저장 (하나의 트랜잭션/커넥션으로 처리) */
export async function saveEventsBulk(events) {
  if (!events?.length) return [];
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const ids = [];
    for (const e of events) {
      ids.push(await insertEvent(e, client)); // ✅ 같은 client 사용
    }
    await client.query("COMMIT");
    return ids;
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  } finally {
    client.release();
  }
}

/** id로 조회: 프론트/소켓이 기대하는 키로 별칭 일치 */
export async function selectEventById(id) {
  const { rows } = await pool.query(
    `
    SELECT
      id,
      event_type AS type,
      meta->>'seat_no'   AS seat_id,
      meta->>'device_id' AS camera_id,
      person_ids,
      confidences,
      duration_sec,
      started_at,
      ended_at,
      created_at AS at,
      meta
    FROM events
    WHERE id = $1
    `,
    [id]
  );
  return rows[0] || null;
}

/** 최근 침입 조회 (created_at 기준, 대소문자 무시) */
export async function selectRecentIntrusions(sinceISO, limit = 200) {
  const { rows } = await pool.query(
    `
    SELECT
      e.id,
      e.event_type AS type,
      e.meta->>'seat_no'   AS seat_id,
      e.meta->>'device_id' AS camera_id,
      e.person_ids,
      e.confidences,
      e.duration_sec,
      e.started_at,
      e.ended_at,
      e.created_at AS at,
      e.meta,
      s.seat_no,
      s.owner_user_id,
      s.name AS seat_name
    FROM events e
    LEFT JOIN seats s
      ON s.seat_no = (e.meta->>'seat_no')::int
    WHERE LOWER(e.event_type) IN ('intrusion','intrusion_started','intrusion_triggered')
      AND e.created_at >= $1
    ORDER BY e.id DESC
    LIMIT $2
    `,
    [sinceISO, Math.min(limit, 200)]
  );
  return rows;
}
