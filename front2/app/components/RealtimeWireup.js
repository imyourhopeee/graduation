"use client";

import { useEffect } from "react";
import { io } from "socket.io-client";

export default function RealtimeWireup() {
  useEffect(() => {
    console.log("[RealtimeWireup] mounted");
    return () => console.log("[RealtimeWireup] unmounted");
  }, []);
  
  useEffect(() => {
    const URL = process.env.NEXT_PUBLIC_EVENT_SERVER_URL || "http://localhost:3002";
    const socket = io(URL, {
      transports: ["websocket"],
      withCredentials: true, // 쿠키/세션 안 쓰면 false로
    });

    socket.on("connected", (p) => console.log("connected:", p));
    socket.on("event:new", (e) => console.log("event:new", e));
    socket.on("log:new", (l) => console.log("log:new", l));

    return () => socket.disconnect();
  }, []);

  return null; // 화면에 렌더 X, 연결만 유지
}
