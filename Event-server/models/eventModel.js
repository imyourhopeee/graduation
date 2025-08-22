// 실제 DB에 데이터를 저장하는 쿼리 로직

// models/eventModel.js (CommonJS)
const pool = require("../db"); // pool.query 사용 (당신 코드와 동일 컨벤션)

// --- 규칙 파라미터 ---
const STAY_SEC = Number(process.env.STAY_SEC || 3);
const DEDUP_WINDOW_SEC = Number(process.env.DEDUP_WINDOW_SEC || 10);

// camera_id -> track_id -> { enterAt, lastSeenAt, zoneId }
const trackState = new Map();
// 이벤트 중복 억제 키 -> lastEmittedAt(sec)
const lastEmit = new Map();

function dedupKey(device_id, track_id, type, zone_id) {
  return `${device_id}:${track_id}:${type}:${zone_id || "-"}`;
}

// === DB I/O ===
async function saveEvent(row, client = pool) {
  const q = `
    INSERT INTO events
      (type, device_id, zone_id, track_id, user_label, started_at, ended_at, duration_sec, meta)
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
    RETURNING id;
  `;
  const v = [
    row.type,
    row.device_id || null,
    row.zone_id || null,
    row.track_id || null,
    row.user_label || null,
    row.started_at || null,
    row.ended_at || null,
    row.duration_sec || null,
    row.meta || {},
  ];
  const { rows } = await client.query(q, v);
  return rows[0].id;
}

async function saveEventsBulk(events) {
  if (!events?.length) return [];
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const ids = [];
    for (const e of events) {
      ids.push(await saveEvent(e, client));
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

// === 규칙 엔진: DetectionBatch -> {persist, realtime} ===
/**
 * @param {object} batch  DetectionBatch(JSON) from AI-server
 * @returns {{ persist: Array<object>, realtime: object }}
 */
function applyRules(batch) {
  const device_id = batch.camera_id;
  const ts = batch.timestamp;

  if (!trackState.has(device_id)) trackState.set(device_id, new Map());
  const camMap = trackState.get(device_id);

  const persist = [];

  for (const det of batch.detections || []) {
    const tid = det.track_id;
    const state = camMap.get(tid) || {
      enterAt: ts,
      lastSeenAt: ts,
      zoneId: null,
    };
    state.lastSeenAt = ts;

    // TODO: ROI 포함 여부로 zoneId 판정 (현재는 null)
    const zoneId = state.zoneId;

    // 체류(Stay) 판단
    const dwellSec = Math.max(
      0,
      (new Date(ts) - new Date(state.enterAt)) / 1000
    );
    if (dwellSec >= STAY_SEC) {
      const k = dedupKey(device_id, tid, "stay", zoneId);
      const nowSec = Date.now() / 1000;
      const last = lastEmit.get(k) || 0;

      if (nowSec - last >= DEDUP_WINDOW_SEC) {
        persist.push({
          type: "stay",
          device_id,
          zone_id: zoneId,
          track_id: tid,
          user_label: det?.face?.label || det?.reid?.densenet_label || null,
          started_at: state.enterAt,
          ended_at: ts,
          duration_sec: Math.floor(dwellSec),
          meta: { bbox: det.bbox, score: det.score },
        });
        lastEmit.set(k, nowSec);
      }
    }

    camMap.set(tid, state);
  }

  const realtime = {
    camera_id: device_id,
    timestamp: ts,
    count: batch.detections?.length || 0,
  };
  return { persist, realtime };
}

module.exports = {
  // 컨트롤러에서 바로 사용
  applyRules,
  saveEvent,
  saveEventsBulk,
};
