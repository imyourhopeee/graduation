import { Router } from "express";
import {
  ingestDetections,
  listEvents,
  addEvent,           
  listIntrusions     
} from "../controllers/eventController.js";
import { verifyAI, requireUser } from "../middleware/authMiddleware.js";

const router = Router();

router.post("/detections", verifyAI, ingestDetections);  
router.post("/events", verifyAI, addEvent);//events 추가함
// router.get("/", listEvents);
router.get("/", requireUser, listEvents);
router.get("/intrusions", requireUser, listIntrusions);

export default router;
