import express from "express";
import http from "http";
import cors from "cors";
import dotenv from "dotenv";
dotenv.config();

import { setupSocket } from "./socket/socket.js";

const app = express();
app.use(
  cors({
    origin: (process.env.CORS_ORIGIN || "*").split(","),
    credentials: true,
  })
);
app.use(express.json({ limit: "5mb" }));

// routes
import authRoutes from "./routes/auth.js";
import eventRoutes from "./routes/events.js";
import logRoutes from "./routes/logs.js";
app.use("/api/v1/auth", authRoutes);
app.use("/api/v1", eventRoutes);
app.use("/api/v1", logRoutes);

// health
app.get("/healthz", (req, res) => res.json({ status: "ok" }));

// socket
const server = http.createServer(app);
const socketApi = setupSocket(server);
app.set("socket", socketApi);

// error handler(최하단)
app.use((err, req, res, next) => {
  console.error("[ERROR]", err);
  res
    .status(err.status || 500)
    .json({ message: err.message || "Internal Error" });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => console.log(`Event-server listening on :${PORT}`));
