// socket/socket.js
import { Server } from "socket.io";

let io = null;

export function setupSocket(server) {
  const origins = (process.env.CORS_ORIGIN || "http://localhost:3000")
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);

  io = new Server(server, {
    cors: {
      origin: origins.length ? origins : ["http://localhost:3000"], // ← origins 사용
      credentials: true,
    },
    path: "/socket.io",
    transports: ["polling", "websocket"],
    upgrade: true,
  });

  io.on("connection", (socket) => {
    console.log("[socket] connected:", socket.id);
    socket.on("hello", (payload) => {
      console.log("[socket] hello:", payload);
      socket.emit("server-event", { ok: true, ts: Date.now() });
    });
    socket.join("dashboard");
    socket.emit("connected", { sid: socket.id });
  });

  return io; // ← 래퍼 객체 말고 io 인스턴스를 직접 리턴
}

export function getIO() {
  if (!io) throw new Error("socket.io not initialized");
  return io;
}
