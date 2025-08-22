import { query } from "./index.js";

async function init() {
  await query(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      username TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      role TEXT DEFAULT 'viewer',
      created_at TIMESTAMPTZ DEFAULT now()
    );
  `);
  await query(`
    CREATE TABLE IF NOT EXISTS events (
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
    );
  `);
  await query(`
    CREATE TABLE IF NOT EXISTS logs (
      id BIGSERIAL PRIMARY KEY,
      level TEXT,
      message TEXT,
      context JSONB,
      created_at TIMESTAMPTZ DEFAULT now()
    );
  `);
  console.log("DB initialized");
  process.exit(0);
}

init().catch((e) => {
  console.error(e);
  process.exit(1);
});
