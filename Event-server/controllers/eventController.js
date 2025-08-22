import { z } from "zod";
import * as db from "../db/queries.js";
import { apply } from "../models/eventModel.js";

const DetectionSchema = z.object({
  camera_id: z.string(),
  timestamp: z.string(),
  fps: z.number().optional(),
  frame_idx: z.number().optional(),
  detections: z
    .array(
      z.object({
        track_id: z.number(),
        bbox: z.array(z.number()).length(4),
        cls: z.string(),
        score: z.number(),
        reid: z
          .object({
            osnet: z.string().nullable().optional(),
            densenet_label: z.string().nullable().optional(),
            densenet_conf: z.number().nullable().optional(),
          })
          .optional(),
        face: z
          .object({
            label: z.string().nullable().optional(),
            conf: z.number().nullable().optional(),
            embedding: z.string().nullable().optional(),
          })
          .optional(),
        center: z.array(z.number()).length(2).optional(),
      })
    )
    .default([]),
});

export async function ingestDetections(req, res, next) {
  try {
    const batch = DetectionSchema.parse(req.body);
    const results = apply(batch);

    const ids = await db.insertEvents(results.persist);

    // 실시간 브로드캐스트
    req.app.get("socket").broadcastEvent({
      ...results.realtime,
      new_event_ids: ids,
    });

    res.json({ accepted: true, new_events: ids });
  } catch (e) {
    next(e);
  }
}

export async function listEvents(req, res, next) {
  try {
    const camera_id = req.query.camera_id || null;
    const limit = Number(req.query.limit || 100);
    const rows = await db.getEvents({ camera_id, limit });
    res.json(rows);
  } catch (e) {
    next(e);
  }
}
