// socket/socket.js
import { Server } from "socket.io";

export function setupSocket(server) {
  const origins = (process.env.CORS_ORIGIN || "http://localhost:3000")
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);

  const io = new Server(server, {
    cors: {
      origin: ["http://localhost:3000"],
      credentials: true,
    },
    path: "/socket.io",
    transports: ["polling", "websocket"],
    upgrade: true,   // 필요하면 강제
  });

  io.on("connection", (socket) => {
    console.log("[socket] connected:", socket.id);
    socket.on("hello", (payload) => {
      console.log("[socket] hello:", payload);
      socket.emit("server-event", { ok: true, ts: Date.now() });
    });
    socket.join("dashboard");
    socket.emit("connected", { sid: socket.id });
    socket.on("disconnect", () => {});
  });

  return {
    io,
    broadcastEvent: (payload) => io.emit("server-event", payload),
    broadcastEvent: (payload) => io.to("dashboard").emit("event:new", payload),
    broadcastLog: (payload) => io.to("dashboard").emit("log:new", payload),
  };
}
