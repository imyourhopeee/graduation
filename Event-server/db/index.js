// DB connection code
import 'dotenv/config';   
import pkg from "pg";
const { Pool } = pkg;

export const pool = new Pool({
  connectionString: process.env.DB_URL,
  max: 10,
});

export async function query(text, params) {
  const res = await pool.query(text, params);
  return res;
}

export async function assertOffeye() {
  const { rows } = await query('SELECT current_database() db, current_user usr');
  if (rows[0].db !== 'offeye') throw new Error(`현재 DB=${rows[0].db}. .env의 DB_URL을 /offeye로 바꿔주세요.`);
  console.log(`[DB] connected to ${rows[0].db} as ${rows[0].usr}`);
}