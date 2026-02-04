//backend\Event-server\routes\events copy.js
import { Router } from "express";
import { verifyAI, requireUser } from "../middleware/authMiddleware.js";
import { ingestDetections, addEvent , notifyEvent} from "../controllers/eventController.js"; //, notifyEvent
import { query } from "../db/index.js";

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
// router.get("/_quick", (req, res) => {
//   return res.json([{ type: "quick_ok", at: Math.floor(Date.now() / 1000) }]);
// });

// AI → 이벤트 생성
router.post("/detections", verifyAI, ingestDetections);
router.post("/", verifyAI, addEvent);
router.post("/notify", verifyAI, notifyEvent);  // DB 저장 없이 즉시 소켓 알림


// 사용자 → 조회 (응답 보장 래퍼)
router.get("/", requireUser, async (req, res, next) => {
  try {
    // 1) 입력 파라미터 정규화
    const rawLimit = Number.parseInt(req.query.limit, 10);
    const limit = Math.max(1, Math.min(Number.isFinite(rawLimit) ? rawLimit : 50, 200));
    const cursor = req.query.cursor ? Number(req.query.cursor) : null;

    // 2) 필요한 컬럼만 선택 (meta 제외)
    //    커서는 id 기준으로 단방향 페이지네이션
    const params = [];
        let sql = `
        SELECT
          id,
          event_type AS type,
          meta->>'seat_no'   AS seat_id,
          meta->>'device_id' AS camera_id,
          person_ids,
          confidences,
          duration_sec,
          started_at,
          ended_at,
          COALESCE(ended_at, started_at, created_at) AS at,
          meta
        FROM events
     `;

    if (!Number.isNaN(cursor) && cursor > 0) {
      sql += ` WHERE id < $1`;
      params.push(cursor);
    }

    // LIMIT 파라미터는 마지막에 바인딩
    const limitPos = params.length + 1;
    sql += ` ORDER BY id DESC LIMIT $${limitPos}`;
    params.push(limit);

    // 3) 실행
    const { rows } = await query(sql, params);

    // 5) 응답
    // 4) 하위호환 매핑: person_id/confidence 키로 배열 내려주기 + meta 정규화
    const mapped = rows.map(r => ({
        ...r,
        person_id: Array.isArray(r.person_ids) ? r.person_ids : (r.person_ids ?? []),
        confidence: Array.isArray(r.confidences) ? r.confidences : (r.confidences ?? []),
        meta: typeof r.meta === "string" ? (()=>{ try { return JSON.parse(r.meta); } catch { return {}; } })() : (r.meta ?? {}),
        }));
    
        // 5) nextCursor 계산 (마지막 행의 id)
    const nextCursor = mapped.length ? mapped[mapped.length - 1].id : null;

        // 6) 응답
    return res.status(200).json({ events: mapped, nextCursor });

  } catch (err) {
    return next(err);
  }
});
// 👇 /events 아래에 추가 (/:id 라우트보다 위에 두기)
router.get("/intrusions", requireUser, async (req, res, next) => {
  try {
    // 1) 파라미터 정규화
    const rawLimit = Number.parseInt(req.query.limit, 10);
    const limit = Math.max(1, Math.min(Number.isFinite(rawLimit) ? rawLimit : 50, 200));

    const rawCursor = Number.parseInt(req.query.cursor, 10);
    const cursor = Number.isFinite(rawCursor) ? rawCursor : null;

    // 선택 필터 (있을 때만 적용)
    const seat = req.query.seat_id ?? req.query.seat;
    const cam  = req.query.camera_id ?? req.query.cam;
    const who  = req.query.person_id ?? req.query.person ?? req.query.identity;
    const since = req.query.since ? new Date(req.query.since) : null; // ISO or yyyy-mm-dd
    const until = req.query.until ? new Date(req.query.until) : null;

    // 2) 조건 구성 (intrusion 타입만)
    const where = [];
    const params = [];

    // cursor: id 단방향 페이지네이션
    if (!Number.isNaN(cursor) && cursor > 0) {
      params.push(cursor);
      where.push(`id < $${params.length}`);
    }

    // intrusion 계열만
    where.push(`LOWER(event_type) IN ('intrusion','intrusion_started','intrusion_triggered')`);

    if (seat) {
      params.push(String(seat));
      where.push(`meta->>'seat_no' = $${params.length}`);
    }
    if (cam) {
      params.push(String(cam));
      where.push(`meta->>'device_id' = $${params.length}`);
    }
    if (who) {
      params.push(JSON.stringify([String(who)]));
      // person_id 또는 identity 를 같은 칼럼에 저장했다면 person_id 기준, 아니라면 필요에 맞게 수정
      where.push(`person_ids @> $${params.length}::jsonb`);
    }
     if (since && !Number.isNaN(since.getTime())) {
       params.push(since);
       where.push(`created_at >= $${params.length}`);
     }
     if (until && !Number.isNaN(until.getTime())) {
       params.push(until);
       where.push(`created_at < $${params.length}`);
     }
     //여길 수정! person_id-> person_ids
     let sql = `
       SELECT
         id,
         event_type AS type,
         meta->>'seat_no'   AS seat_id,
         meta->>'device_id' AS camera_id,
         person_ids,
         confidences,
         duration_sec,
         started_at,
         ended_at,
         COALESCE(ended_at, started_at, created_at) AS at,
         meta
       FROM events
       ${where.length ? `WHERE ${where.join(" AND ")}` : ""}
       ORDER BY id DESC
     `;

    // LIMIT 마지막에 바인딩
    params.push(limit);
    sql += ` LIMIT $${params.length}`;

    const { rows } = await query(sql, params);
    const mapped = rows.map(r => ({
      ...r,
      person_id: Array.isArray(r.person_ids) ? r.person_ids : (r.person_ids ?? []),
      confidence: Array.isArray(r.confidences) ? r.confidences : (r.confidences ?? []),
      meta: typeof r.meta === "string" ? (()=>{ try { return JSON.parse(r.meta); } catch { return {}; } })() : (r.meta ?? {}),
    }));
    const nextCursor = mapped.length ? mapped[mapped.length - 1].id : null;
    return res.status(200).json({ events: mapped, nextCursor });
  } catch (err) {
    return next(err);
  }
});

router.get("/:id", requireUser, async (req, res, next) => {
  try {
    const id = Number(req.params.id);
    if (!Number.isFinite(id) || id <= 0) return res.status(400).json({ error: "bad id" });

   const { rows } = await query(
     `SELECT
        id,
        event_type AS type,
        meta->>'seat_no'   AS seat_id,
        meta->>'device_id' AS camera_id,
        person_ids,
        confidences,
        duration_sec,
        started_at,
        ended_at,
        COALESCE(ended_at, started_at, created_at) AS at,
        meta
      FROM events 
      WHERE id = $1`,
     [id] ///1027 , 수정
   );
    if (!rows.length) return res.status(404).json({ error: "not found" });
    const r = rows[0];
    return res.json({
      ...r,
      person_id: Array.isArray(r.person_ids) ? r.person_ids : (r.person_ids ?? []),
      confidence: Array.isArray(r.confidences) ? r.confidences : (r.confidences ?? []),
      meta: typeof r.meta === "string" ? (()=>{ try { return JSON.parse(r.meta); } catch { return {}; } })() : (r.meta ?? {}),
    });
  } catch (e) {
    return next(e);
  }
});

export default router;
