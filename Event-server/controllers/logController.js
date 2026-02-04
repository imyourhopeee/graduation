import { insertLog } from "../db/queries.js";

export async function writeLog(req, res, next) {
  try {
    const id = await insertLog({
      level: req.body.level || "info",
      message: req.body.message || "",
      context: req.body.context || {},
    });
    req.app
      .get("socket")
      .broadcastLog({ id, ...req.body, created_at: new Date().toISOString() });
    res.status(201).json({ id });
  } catch (e) {
    next(e);
  }
}
