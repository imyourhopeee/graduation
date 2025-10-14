import { Router } from "express";
// import {
//   ingestDetections,
//   listEvents,
//   addEvent,
//   listIntrusions,
// } from "../controllers/eventController.js";
import { verifyAI, requireUser } from "../middleware/authMiddleware.js";
import { ingestDetections, addEvent } from "../controllers/eventController.js";
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

// 사용자 → 조회 (응답 보장 래퍼)
router.get("/", requireUser, async (req, res, next) => {
  try {
    // 1) 입력 파라미터 정규화
    const rawLimit = Number(req.query.limit ?? 50);
    const limit = Math.max(1, Math.min(rawLimit, 200)); // 상한 200
    const cursor = req.query.cursor ? Number(req.query.cursor) : null;

    // 2) 필요한 컬럼만 선택 (meta 제외)
    //    커서는 id 기준으로 단방향 페이지네이션
    const params = [];
    let sql = `
      SELECT
        id,
        event_type,
        seat_id,
        camera_id,
        person_id,
        confidence,
        started_at,
        ended_at,
        created_at
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

    // 4) nextCursor 계산 (마지막 행의 id)
    const nextCursor = rows.length ? rows[rows.length - 1].id : null;

    // 5) 응답
    return res.status(200).json({ events: rows, nextCursor });
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
      where.push(`seat_id = $${params.length}`);
    }
    if (cam) {
      params.push(String(cam));
      where.push(`camera_id = $${params.length}`);
    }
    if (who) {
      params.push(String(who));
      // person_id 또는 identity 를 같은 칼럼에 저장했다면 person_id 기준, 아니라면 필요에 맞게 수정
      where.push(`person_id = $${params.length}`);
    }
    if (since && !Number.isNaN(since.getTime())) {
      params.push(since);
      // at/ended_at/started_at 중 at 기준 (스키마에 맞게 조정 가능)
      where.push(`at >= $${params.length}`);
    }
    if (until && !Number.isNaN(until.getTime())) {
      params.push(until);
      where.push(`at < $${params.length}`);
    }

    let sql = `
      SELECT
        id,
        event_type,
        seat_id,
        camera_id,
        person_id,
        confidence,
        started_at,
        ended_at,
        created_at
      FROM events
      ${where.length ? `WHERE ${where.join(" AND ")}` : ""}
      ORDER BY id DESC
    `;

    // LIMIT 마지막에 바인딩
    params.push(limit);
    sql += ` LIMIT $${params.length}`;

    const { rows } = await query(sql, params);
    const nextCursor = rows.length ? rows[rows.length - 1].id : null;

    return res.status(200).json({ events: rows, nextCursor });
  } catch (err) {
    return next(err);
  }
});

router.get("/:id", requireUser, async (req, res, next) => {
  try {
    const id = Number(req.params.id);
    if (!Number.isFinite(id) || id <= 0) return res.status(400).json({ error: "bad id" });

    const { rows } = await query(
      `SELECT id, event_type, seat_id, camera_id, person_id, confidence,
              started_at, ended_at, created_at, meta
       FROM events WHERE id = $1`,
      [id]
    );
    if (!rows.length) return res.status(404).json({ error: "not found" });
    return res.json(rows[0]);
  } catch (e) {
    return next(e);
  }
});

export default router;
