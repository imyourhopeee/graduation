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
