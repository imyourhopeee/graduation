import { Router } from "express";
import {
  ingestDetections,
  listEvents,
  addEvent,
  listIntrusions,
} from "../controllers/eventController.js";
import { verifyAI, requireUser } from "../middleware/authMiddleware.js";

const router = Router();

// IN/OUT 로깅
router.use((req, res, next) => {
  const t0 = Date.now();
  console.log(`[events] IN  ${req.method} ${req.originalUrl}`);
  res.on("finish", () => {
    console.log(
      `[events] OUT ${req.method} ${req.originalUrl} -> ${res.statusCode} ${Date.now() - t0}ms`
    );
  });
  next();
});

// 헬스체크 & 빠른 확인용
router.get("/healthz", (req, res) => res.json({ ok: true, now: Date.now() }));
router.get("/_quick", (req, res) => {
  return res.json([{ type: "quick_ok", at: Math.floor(Date.now() / 1000) }]);
});

// AI → 이벤트 생성
router.post("/detections", verifyAI, ingestDetections);
router.post("/", verifyAI, addEvent);

// 사용자 → 조회 (응답 보장 래퍼)
router.get("/", requireUser, async (req, res, next) => {
  console.log("[DEBUG] requireUser 통과됨");
  try {
    const out = await listEvents(req, res, next);
    if (!res.headersSent) {
      return res.status(200).json(out ?? []);
    }
    console.log("[DEBUG] listEvents 종료 (controller sent)");
    return; // 명시적으로 종료
  } catch (err) {
    console.error("[DEBUG] listEvents 예외:", err);
    // 에러 핸들러로 위임 (여기서 응답 보내지 않음)
    return next(err);
  }
});

// 침입 전용 조회 (동일 패턴)
router.get("/intrusions", requireUser, async (req, res, next) => {
  try {
    const out = await listIntrusions(req, res, next);
    if (!res.headersSent) {
      return res.status(200).json(out ?? []);
    }
    return;
  } catch (e) {
    return next(e);
  }
});

export default router;
