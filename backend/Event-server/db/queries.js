// db/querys.js
import { query } from "./index.js";

export async function insertEvents(events) {
  if (!events?.length) return [];
  const ids = [];
  for (const e of events) {
    // DB 컬럼에 맞게 매핑: event_type/started_at/ended_at/duration_sec/person_id/confidence/meta
    const event_type = (e.event_type ?? e.type ?? "").toString().toLowerCase() || null;

    // meta JSONB에 비컬럼 값들을 수렴 (device_id/seat_no(user_label/track_id 포함))
    const meta = {
      ...(e.meta || {}),
      ...(e.device_id != null ? { device_id: String(e.device_id) } : {}),
      ...(e.zone_id != null ? { seat_no: Number(e.zone_id) } : {}), // 좌석 번호는 seat_no로
      ...(e.user_label != null ? { user_label: String(e.user_label) } : {}),
      ...(e.track_id != null ? { track_id: String(e.track_id) } : {}),
    };

    const params = [
      event_type,                  // $1
      e.started_at || null,        // $2
      e.ended_at || null,          // $3
      e.duration_sec ?? null,      // $4
      e.person_id ?? null,         // $5
      e.confidence ?? null,        // $6
      meta,                        // $7 (JSONB)
    ];

    const { rows } = await query(
      `INSERT INTO events
         (event_type, started_at, ended_at, duration_sec, person_id, confidence, meta)
       VALUES ($1,$2,$3,$4,$5,$6,$7)
       RETURNING id`,
      params
    );
    ids.push(rows[0].id);
  }
  return ids;
}

export async function getEvents({ camera_id, limit = 100 }) {
  const lim = Number.isFinite(Number(limit)) ? Math.max(1, Math.min(Number(limit), 500)) : 100;
  const { rows } = await query(
    `SELECT *
       FROM events
      WHERE ($1::text IS NULL OR meta->>'device_id' = $1)
      ORDER BY id DESC
      LIMIT $2`,
    [camera_id || null, lim]
  );
  return rows;
}

export async function insertLog({ level = "info", message, context = {} }) {
  const { rows } = await query(
    `INSERT INTO logs (level, message, context) VALUES ($1,$2,$3) RETURNING id`,
    [level, message, context]
  );
  return rows[0].id;
}
