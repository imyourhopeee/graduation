
import 'dotenv/config'; 
import { query, pool } from "./index.js";

const DDLS = [
  // 확장 (corr_id용 uuid 생성 등)
  `CREATE EXTENSION IF NOT EXISTS pgcrypto;`,

  // 기존 테이블
  `CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'viewer',
    created_at TIMESTAMPTZ DEFAULT now()
  );`,

  `CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    type TEXT,
    device_id TEXT,
    zone_id TEXT,
    track_id INT,
    user_label TEXT,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    duration_sec INT,
    meta JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
  );`,

  `CREATE TABLE IF NOT EXISTS logs (
    id BIGSERIAL PRIMARY KEY,
    level TEXT,
    message TEXT,
    context JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
  );`,

  // ✅ 최소 변경: events 컬럼 보강
  `ALTER TABLE events
     ADD COLUMN IF NOT EXISTS event_type   TEXT,
     ADD COLUMN IF NOT EXISTS correlation_id UUID,
     ADD COLUMN IF NOT EXISTS person_id    TEXT,
     ADD COLUMN IF NOT EXISTS status       TEXT,
     ADD COLUMN IF NOT EXISTS confidence   NUMERIC;`,
  // --- users 보강: 컬럼/디폴트/제약 ---
  `ALTER TABLE users
     ADD COLUMN IF NOT EXISTS name        TEXT,
     ADD COLUMN IF NOT EXISTS updated_at  TIMESTAMPTZ DEFAULT now();`,

  // -- gen_random_uuid() 디폴트 (pgcrypto 확장 필요, 이미 상단에서 생성)
  `ALTER TABLE users
     ALTER COLUMN id SET DEFAULT gen_random_uuid();`,

  // -- role NOT NULL 보강 (원하면)
  `ALTER TABLE users
     ALTER COLUMN role SET NOT NULL;`,

  // 인덱스
  `CREATE INDEX IF NOT EXISTS idx_events_corr     ON events(correlation_id);`,
  `CREATE INDEX IF NOT EXISTS idx_events_type     ON events(event_type);`,
  `CREATE INDEX IF NOT EXISTS idx_events_created  ON events(created_at);`,
  `CREATE INDEX IF NOT EXISTS idx_events_zone     ON events(zone_id);`,
  `CREATE INDEX IF NOT EXISTS idx_events_device   ON events(device_id);`,
  // 메타검색이 잦으면 GIN도 고려:
  // `CREATE INDEX IF NOT EXISTS idx_events_meta ON events USING GIN (meta);`,
];

async function init() {
  // 🔒 안전: 현재 접속 DB가 offeye인지 확인
  const { rows } = await query(`SELECT current_database() AS db, current_user AS usr`);
  if (rows[0].db !== "offeye") {
    throw new Error(`지금 ${rows[0].db} DB에 연결되어 있어요. .env의 DATABASE_URL을 offeye로 바꿔주세요.`);
  }
  console.log(`[init] connected to db=${rows[0].db} as user=${rows[0].usr}`);

  try {
    await query("BEGIN");
    for (const sql of DDLS) {
      await query(sql);
    }
    await query("COMMIT");
    console.log("DB initialized ✅");
  } catch (e) {
    await query("ROLLBACK");
    console.error("DB init failed ❌", e);
    process.exit(1);
  } finally {
    await pool.end();
  }
}

init();
