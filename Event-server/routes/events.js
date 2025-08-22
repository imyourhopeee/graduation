import { Router } from "express";
import {
  ingestDetections,
  listEvents,
} from "../controllers/eventController.js";
import { verifyAI, requireUser } from "../middleware/authMiddleware.js";

const router = Router();

router.post("/detections", verifyAI, ingestDetections); // AI-server → Event-server
router.get("/events", requireUser, listEvents); // Frontend → Event-server

export default router;
