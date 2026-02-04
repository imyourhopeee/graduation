import { Router } from "express";
import { writeLog } from "../controllers/logController.js";
import { requireUser } from "../middleware/authMiddleware.js"; 


const router = Router();
router.post("/", requireUser, writeLog); /* /logs 일때 오류 생김 */

export default router;
