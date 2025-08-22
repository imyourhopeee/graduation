import { Router } from "express";
import { writeLog } from "../controllers/logController.js";
import { requireUser } from "../middleware/authMiddleware.js";

const router = Router();
router.post("/logs", requireUser, writeLog);

export default router;
