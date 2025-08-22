import { query } from "./index.js";

export async function insertEvents(events) {
  if (!events?.length) return [];
  const ids = [];
  for (const e of events) {
    const { rows } = await query(
      `INSERT INTO events
        (type, device_id, zone_id, track_id, user_label, started_at, ended_at, duration_sec, meta)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
       RETURNING id`,
      [
        e.type,
        e.device_id || null,
        e.zone_id || null,
        e.track_id || null,
        e.user_label || null,
        e.started_at || null,
        e.ended_at || null,
        e.duration_sec || null,
        e.meta || {},
      ]
    );
    ids.push(rows[0].id);
  }
  return ids;
}

export async function getEvents({ camera_id, limit = 100 }) {
  const { rows } = await query(
    `SELECT * FROM events
     WHERE ($1::text IS NULL OR device_id=$1)
     ORDER BY id DESC
     LIMIT $2`,
    [camera_id || null, limit]
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
