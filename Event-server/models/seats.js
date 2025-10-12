import { query } from "../db/index.js";

export async function listSeats() {
  const { rows } = await query(
    `SELECT id, seat_no, owner_user_id, name, config, created_at
     FROM seats ORDER BY seat_no ASC`
  );
  return rows;
}

export async function upsertSeat({ seat_no, owner_user_id=null, name=null, config=null }) {
  const { rows } = await query(
    `INSERT INTO seats (seat_no, owner_user_id, name, config)
     VALUES ($1,$2,$3,$4)
     ON CONFLICT (seat_no) DO UPDATE
       SET owner_user_id=EXCLUDED.owner_user_id,
           name=EXCLUDED.name,
           config=EXCLUDED.config
     RETURNING id, seat_no, owner_user_id, name, config, created_at`,
    [seat_no, owner_user_id, name, config]
  );
  return rows[0];
}
