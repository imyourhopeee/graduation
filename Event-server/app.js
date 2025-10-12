//app.js
import dotenv from "dotenv";
dotenv.config();
import express from "express";
import http from "http";
import cors from "cors";
// import cookieParser from "cookie-parser";
import { assertOffeye } from './db/index.js'; 
import { setupSocket } from "./socket/socket.js";

const app = express();
// app.js 맨 위쪽 미들웨어들 전에 추가함!
app.use((req, res, next) => {
  console.log(`[REQ] ${req.method} ${req.originalUrl}`);
  next();
});

// app.use(cookieParser()); //추가함 
// CORS
const origins = (process.env.CORS_ORIGIN || '')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

app.use(
  cors({
    origin: origins.length ? origins : ["http://localhost:3000"],
    methods: ["GET","POST","PUT","DELETE","OPTIONS"], //추가
    allowedHeaders: ["Content-Type", "Authorization"], //추가
    credentials: true,
  })
);

app.use(express.json({ limit: "5mb" }));

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
    .json({ message: err.message || "Internal Error: 서버에 오류가 발생했습니다." });
});


// socket
const PORT = Number(process.env.PORT || 3002);
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
    console.error('[BOOT] failed to assert DB', e);
    process.exit(1);
  }
})();

console.log("[boot] PORT=", process.env.PORT, "AI_JWT_SECRET length=", (process.env.AI_JWT_SECRET||"").length);
console.log("[boot] cwd=", process.cwd());

