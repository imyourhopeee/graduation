// //events.js
// routes/events.js
import { Router } from "express";
import {
  ingestDetections,
  listEvents,
  addEvent,
  listIntrusions
} from "../controllers/eventController.js";
import { verifyAI, requireUser } from "../middleware/authMiddleware.js";

const router = Router();

// ✅ 추가: 라우터 단위 IN/OUT 로깅(응답이 닫히는지 확인)
router.use((req, res, next) => {
  const t0 = Date.now();
  console.log(`[events] IN  ${req.method} ${req.originalUrl}`);
  res.on("finish", () => {
    console.log(`[events] OUT ${req.method} ${req.originalUrl} -> ${res.statusCode} ${Date.now() - t0}ms`);
  });
  next();
});

// ✅ 추가: 헬스 체크(라우팅 자체가 즉시 200 되는지 확인)
router.get("/healthz", (req, res) => res.json({ ok: true, now: Date.now() }));

// ✅ 추가: 빠른 우회 엔드포인트(인증 없이 바로 응답) — 병목 위치 분리용
router.get("/_quick", (req, res) => {
  res.json([{ type: "quick_ok", at: Math.floor(Date.now() / 1000) }]);
});

// ✅ 추가(선택): 퍼블릭 조회(디버그 시만; 운영 시 제거하세요)
router.get("/public", listEvents);

router.post("/detections", verifyAI, ingestDetections);
router.post("/", verifyAI, addEvent);

// router.get("/", listEvents);

// // ✅ 추가: requireUser 통과/DB 응답 여부를 로그로 정확히 찍기
// router.get("/", requireUser, async (req, res, next) => {
//   console.log("[DEBUG] requireUser 통과됨");        // ← 여기까지 오면 토큰 검증은 OK
//   try {
//     await listEvents(req, res, next);               // ← 응답이 정상적으로 닫혀야 함
//     console.log("[DEBUG] listEvents 종료");         // ← 이 로그가 찍히면 DB/핸들러 완료
//   } catch (err) {
//     console.error("[DEBUG] listEvents 예외:", err); // ← 에러 경로도 명확히 찍기
//     next(err);
//   }
// });
router.get("/", requireUser, async (req, res, next) => {
  console.log("[DEBUG] requireUser 통과됨");
  try {
    const out = await listEvents(req, res, next); // 컨트롤러가 직접 res.json 할 수도, 값을 return 할 수도 있음

    // 컨트롤러가 직접 보냈다면(headersSent=true) 아무 것도 하지 않음
    if (!res.headersSent) {
      // 컨트롤러가 값을 리턴만 했다면 여기서 보냄
      res.status(200).json(out ?? []);
    }

    console.log("[DEBUG] listEvents 종료");
  } catch (err) {
    console.error("[DEBUG] listEvents 예외:", err);
    // 에러 응답도 반드시 닫기
    if (!res.headersSent) res.status(500).json({ message: "internal error" });
    next(err);
  }
});

// (선택) 동일 문제 방지
router.get("/intrusions", requireUser, async (req, res, next) => {
  try {
    const out = await listIntrusions(req, res, next);
    if (!res.headersSent) res.status(200).json(out ?? []);
  } catch (e) {
    if (!res.headersSent) res.status(500).json({ message: "internal error" });
    next(e);
  }
});

export default router;








// import { Router } from "express";
// import {
//   ingestDetections,
//   listEvents,
//   addEvent,           
//   listIntrusions     
// } from "../controllers/eventController.js";
// import { verifyAI, requireUser } from "../middleware/authMiddleware.js";

// const router = Router();

// // ✅ 추가: 라우터 단위 IN/OUT 로깅(응답이 닫히는지 확인)
// router.use((req, res, next) => {
//   const t0 = Date.now();
//   console.log(`[events] IN  ${req.method} ${req.originalUrl}`);
//   res.on("finish", () => {
//     console.log(`[events] OUT ${req.method} ${req.originalUrl} -> ${res.statusCode} ${Date.now() - t0}ms`);
//   });
//   next();
// });

// // ✅ 추가: 헬스 체크(라우팅 자체가 즉시 200 되는지 확인)
// router.get("/healthz", (req, res) => res.json({ ok: true, now: Date.now() }));

// // ✅ 추가: 빠른 우회 엔드포인트(인증 없이 바로 응답) — 병목 위치 분리용
// router.get("/_quick", (req, res) => {
//   res.json([{ type: "quick_ok", at: Math.floor(Date.now() / 1000) }]);
// });

// router.post("/detections", verifyAI, ingestDetections);  
// router.post("/", verifyAI, addEvent);//events 추가함

// // router.get("/", listEvents);
// // 다시 주석해제: router.get("/", requireUser, listEvents);

// // ✅ 교체: GET /events 래퍼에서 "응답 보장" 처리
// router.get("/", requireUser, async (req, res, next) => {
//   console.log("[DEBUG] requireUser 통과됨");
//   try {
//     const out = await listEvents(req, res, next); // 컨트롤러가 직접 res.json 할 수도, 값을 return 할 수도 있음

//     // 컨트롤러가 직접 보냈다면(headersSent=true) 아무 것도 하지 않음
//     if (!res.headersSent) {
//       // 컨트롤러가 값을 리턴만 했다면 여기서 보냄
//       res.status(200).json(out ?? []);
//     }

//     console.log("[DEBUG] listEvents 종료");
//   } catch (err) {
//     console.error("[DEBUG] listEvents 예외:", err);
//     // 에러 응답도 반드시 닫기
//     if (!res.headersSent) res.status(500).json({ message: "internal error" });
//     next(err);
//   }
// });

// // 추후 변경시 주석해제: router.get("/intrusions", requireUser, listIntrusions);
// // (선택) 동일 문제 방지
// router.get("/intrusions", requireUser, async (req, res, next) => {
//   try {
//     const out = await listIntrusions(req, res, next);
//     if (!res.headersSent) res.status(200).json(out ?? []);
//   } catch (e) {
//     if (!res.headersSent) res.status(500).json({ message: "internal error" });
//     next(e);
//   }
// });



// export default router;
