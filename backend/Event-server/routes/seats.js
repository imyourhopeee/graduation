// event-server/routes/seats.js
import { Router } from "express";
import { z } from "zod";
import { listSeats, upsertSeat } from "../models/seats.js";

const router = Router();

router.get("/", async (_req, res) => {
  const rows = await listSeats();
  res.json({ ok:true, seats: rows });
});

const upsertSchema = z.object({
  owner_user_id: z.string().nullable().optional(),
  name: z.string().nullable().optional(),
  config: z.any().nullable().optional()
});

router.put("/:seat_no", async (req, res) => {
  const seat_no = Number(req.params.seat_no);
  if (!Number.isInteger(seat_no)) return res.status(400).json({ ok:false, error:"invalid_seat_no" });

  const parsed = upsertSchema.safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ ok:false, error:"invalid_body" });

  const row = await upsertSeat({ seat_no, ...parsed.data });
  res.json({ ok:true, seat: row });
});

export default router;
