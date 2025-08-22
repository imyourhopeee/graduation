import { Server } from "socket.io";

export function setupSocket(server) {
  const io = new Server(server, {
    cors: {
      origin: (process.env.CORS_ORIGIN || "*").split(","),
      credentials: true,
    },
  });

  io.on("connection", (socket) => {
    socket.join("dashboard");
    socket.emit("connected", { sid: socket.id });
    socket.on("disconnect", () => {});
  });

  return {
    io,
    broadcastEvent: (payload) => io.to("dashboard").emit("event:new", payload),
    broadcastLog: (payload) => io.to("dashboard").emit("log:new", payload),
  };
}
