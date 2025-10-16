"use client";

import { useState, useEffect } from "react";
import Navbar from "../components/Navbar";

export default function SettingPage() {
  const [detectionTime, setDetectionTime] = useState("60"); // 기본값: 60초

  // ✅ 최초 로딩 시 서버에서 현재 dwell 시간 불러오기
  useEffect(() => {
    fetch("http://localhost:3001/config/seats/dwell")
      .then((res) => res.json())
      .then((data) => {
        if (data?.seconds) {
          setDetectionTime(String(Math.round(data.seconds)));
        }
      })
      .catch((err) => {
        console.warn("dwell 불러오기 실패:", err);
      });
  }, []);

  const handleSave = async () => {
    try {
      const res = await fetch("http://localhost:3001/config/seats/dwell", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ seconds: Number(detectionTime) }),
      });

      if (!res.ok) throw new Error("서버 응답 오류");

      alert("설정이 저장되었습니다.");
    } catch (err) {
      console.error("설정 저장 실패:", err);
      alert("서버와 연결할 수 없습니다.");
    }
  };

  return (
    <main className="min-h-screen bg-gray-100 flex flex-col">
      <Navbar />

      <section className="flex-grow flex flex-col items-center p-8 gap-6">
        <h1 className="text-3xl font-bold text-gray-800">⚙️ 시스템 설정</h1>

        <div className="w-full max-w-3xl bg-white rounded-xl shadow p-6 space-y-6">
          {/* ✅ 자리 침입 시간 설정 */}
          <div className="flex flex-col">
            <label className="text-gray-700 font-medium mb-2">
              타인 감지 시간 (초)
            </label>
            <select
              value={detectionTime}
              onChange={(e) => setDetectionTime(e.target.value)}
              className="border px-4 py-2 rounded-lg text-gray-800"
            >
              <option value="5">5초</option>
              <option value="10">10초</option>
              <option value="20">20초</option>
              <option value="30">30초</option>
              <option value="40">40초</option>
              <option value="50">50초</option>
              <option value="60">1분</option>
              <option value="180">3분</option>
              <option value="300">5분</option>
              <option value="600">10분</option>
            </select>
          </div>

          <button
            onClick={handleSave}
            className="bg-mint hover:bg-green text-white px-6 py-3 rounded-xl font-semibold shadow"
          >
            저장하기
          </button>
        </div>
      </section>
    </main>
  );
}
