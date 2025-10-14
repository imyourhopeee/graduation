// DB connection code
import 'dotenv/config';   
import pkg from "pg";
const { Pool } = pkg;

export const pool = new Pool({
  connectionString: process.env.DB_URL,
  max: 10,
});

// 풀 에러 핸들해서 프로세스 크래시 방지
pool.on("error", (err) => {
  console.error("[PG POOL ERROR]", err?.code, err?.message);

});

// 공용 쿼리: 재시도 래퍼 (ECONNRESET, 57P01 등 일시 오류 대응)
const RETRYABLE_CODES = new Set([
  "ECONNRESET", "ECONNREFUSED", "ETIMEDOUT",
  "57P01", // admin_shutdown
  "57P02", // crash_shutdown
  "57P03", // cannot_connect_now
]);

export async function query(text, params, { retries = 3 } = {}) {
  let attempt = 0;
  while (true) {
    try {
      return await pool.query(text, params);
    } catch (e) {
      attempt++;
      if (!RETRYABLE_CODES.has(e.code) || attempt > retries) {
        throw e;
      }
      const backoff = Math.min(1000 * attempt, 3000);
      console.warn(`[PG RETRY ${attempt}/${retries}]`, e.code, e.message);
      await new Promise((r) => setTimeout(r, backoff));
    }
  }
}

export async function assertOffeye() {
  const { rows } = await query('SELECT current_database() db, current_user usr');
  if (rows[0].db !== 'offeye') throw new Error(`현재 DB=${rows[0].db}. .env의 DB_URL을 /offeye로 바꿔주세요.`);
  console.log(`[DB] connected to ${rows[0].db} as ${rows[0].usr}`);
}