//app.js
import dotenv from "dotenv";
dotenv.config();

import express from "express";
import http from "http";
import cors from "cors";
// import cookieParser from "cookie-parser";
import { assertOffeye } from "./db/index.js";
import { query } from "./db/index.js";

import { setupSocket } from "./socket/socket.js";
import pg from "pg";
const { Pool } = pg;
const pool = new Pool({ connectionString: process.env.DB_URL });

const app = express();

async function waitForDb(tries = 10) {
  for (let i = 1; i <= tries; i++) {
    try {
      await query("SELECT 1");
      console.log("[BOOT] DB ready");
      return;
    } catch (e) {
      console.warn(`[BOOT] DB not ready (${i}/${tries})`, e.code || e.message);
      await new Promise((r) => setTimeout(r, Math.min(1000 * i, 5000)));
    }
  }
  throw new Error("DB not ready after retries");
}
const PORT = Number(process.env.PORT || 3002);

try {
  await waitForDb();
  // 라우터/소켓 초기화들...
  const server = http.createServer(app);
  server.listen(PORT, () => console.log(`[BOOT] Event-server on :${PORT}`));
} catch (e) {
  console.error("[BOOT] failed to assert DB", e);
  process.exit(1);
}

// 프로세스 전역 안전망 (로그만 남기고 계속 살려둘 수도)
process.on("unhandledRejection", (reason) => {
  console.error("[UNHANDLED REJECTION]", reason);
});
process.on("uncaughtException", (err) => {
  console.error("[UNCAUGHT EXCEPTION]", err);
})(
  // 여기까지 추가
  async () => {
    try {
      const r = await pool.query(
        "select current_database() db, current_schema() sch, now() now",
      );
      console.log("[DB] url=", process.env.DB_URL);
      console.log(
        "[DB] connect ok db=%s schema=%s now=%s",
        r.rows[0].db,
        r.rows[0].sch,
        r.rows[0].now,
      );
    } catch (e) {
      console.error("[DB] connect fail", e);
    }
  },
)();

// app.js 맨 위쪽 미들웨어들 전에 추가함!
app.use((req, res, next) => {
  console.log(`[REQ] ${req.method} ${req.originalUrl}`);
  next();
});

// app.use(cookieParser()); //추가함
// CORS
const origins = (process.env.CORS_ORIGIN || "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

app.use(
  cors({
    origin: origins.length ? origins : ["http://localhost:3000"],
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"], //추가
    allowedHeaders: ["Content-Type", "Authorization"], //추가
    credentials: true,
  }),
);

app.use(express.json({ limit: "5mb" }));

app.use((req, res, next) => {
  res.set("Cache-Control", "no-store");
  next();
});

// health
app.get("/healthz", (req, res) => res.json({ status: "ok" }));

// routes
import authRoutes from "./routes/auth.js";
import eventRoutes from "./routes/events.js";
import logRoutes from "./routes/logs.js";
app.use("/auth", authRoutes);
app.use("/events", eventRoutes);
app.use("/logs", logRoutes);
app.post("/_debug/emit", (req, res) => {
  const sock = req.app.get("socket");
  sock?.broadcastEvent?.({ msg: "hello from server", ts: Date.now() });
  res.json({ ok: true });
});

app.use((req, res) => res.status(404).json({ message: "Not Found" }));

// error handler(최하단)
app.use((err, req, res, next) => {
  console.error("[ERROR]", err);
  res
    .status(err.status || 500)
    .json({
      message: err.message || "Internal Error: 서버에 오류가 발생했습니다.",
    });
});

// socket

const server = http.createServer(app);
const socketApi = setupSocket(server);
app.set("socket", socketApi);
(async () => {
  try {
    await assertOffeye(); // DB 연결, 현재 DB 이름 확인
    server.listen(PORT, () => {
      console.log(`[event-server] listening on http://localhost:${PORT}`);
    });
  } catch (e) {
    console.error("[BOOT] failed to assert DB", e);
    process.exit(1);
  }
})();

console.log(
  "[boot] PORT=",
  process.env.PORT,
  "AI_JWT_SECRET length=",
  (process.env.AI_JWT_SECRET || "").length,
);
console.log("[boot] cwd=", process.cwd());
