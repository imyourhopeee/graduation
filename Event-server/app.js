// app.js
import dotenv from "dotenv";
dotenv.config();

import express from "express";
import http from "http";
import cors from "cors";
// import cookieParser from "cookie-parser";
import { query, assertOffeye } from "./db/index.js"; // ✅ 한 군데로 통일
import { setupSocket } from "./socket/socket.js";

const app = express();
const PORT = Number(process.env.PORT || 3002);

// ===== 부팅시 DB 준비 대기 =====
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

// ===== 공통 미들웨어 =====
app.use((req, res, next) => {
  console.log(`[REQ] ${req.method} ${req.originalUrl}`);
  next();
});

// app.use(cookieParser());

// CORS
const origins = (process.env.CORS_ORIGIN || "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

app.use(
  cors({
    origin: origins.length ? origins : ["http://localhost:3000"],
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
    credentials: true,
  })
);

app.use(express.json({ limit: "5mb" }));

// no-store
app.use((req, res, next) => {
  res.set("Cache-Control", "no-store");
  next();
});

// health
app.get("/healthz", (req, res) => res.json({ status: "ok" }));

// ===== 라우트 등록 =====
import authRoutes from "./routes/auth.js";
import eventRoutes from "./routes/events.js";
import logRoutes from "./routes/logs.js";

app.use("/auth", authRoutes);
app.use("/events", eventRoutes);
app.use("/logs", logRoutes);

// debug emit
app.post("/_debug/emit", (req, res) => {
  const sock = req.app.get("socket");
  sock?.broadcastEvent?.({ msg: "hello from server", ts: Date.now() });
  res.json({ ok: true });
});

// 404
app.use((req, res) => res.status(404).json({ message: "Not Found" }));

// 에러 핸들러(최하단)
app.use((err, req, res, next) => {
  console.error("[ERROR]", err);
  res
    .status(err.status || 500)
    .json({ message: err.message || "Internal Error: 서버에 오류가 발생했습니다." });
});

// ===== 소켓 + 서버 생성은 한 번만 =====
const server = http.createServer(app);
const io = setupSocket(server);   // ← io 인스턴스
app.set("io", io); 

// ===== 부팅 시퀀스: DB 준비 → 스키마 보장 → 서버 listen =====
try {
  await waitForDb();
  await assertOffeye(); // DB 초기 스키마/준비 확인
  server.listen(PORT, () => {
    console.log(`[event-server] listening on http://localhost:${PORT}`);
  });
} catch (e) {
  console.error("[BOOT] failed to assert DB", e);
  process.exit(1);
}

// ===== 프로세스 전역 안전망 =====
process.on("unhandledRejection", (reason) => {
  console.error("[UNHANDLED REJECTION]", reason);
});
process.on("uncaughtException", (err) => {
  console.error("[UNCAUGHT EXCEPTION]", err);
});

console.log(
  "[boot] PORT=",
  process.env.PORT,
  "AI_JWT_SECRET length=",
  (process.env.AI_JWT_SECRET || "").length
);
console.log("[boot] cwd=", process.cwd());
