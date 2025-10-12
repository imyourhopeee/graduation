"use client";

import { useEffect, useState } from "react";
import Navbar from "../components/Navbar";

export default function EventsPage() {
  const [logs, setLogs] = useState([]);
  const [filteredLogs, setFilteredLogs] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState("전체");

  // ✅ 로그 가져오기 (AI 서버에서 fetch)
  const fetchLogs = async () => {
    try {
      const res = await fetch("http://localhost:3002/routes/logs");
      const data = await res.json();
      setLogs(data.logs || []);
    } catch (err) {
      console.error("이벤트 로그 불러오기 실패", err);
    }
  };

  // ✅ 최초 실행 + 5초마다 polling
  useEffect(() => {
    fetchLogs(); // 최초 실행
    const interval = setInterval(fetchLogs, 5000); // 5초마다

    return () => clearInterval(interval); // 언마운트 시 정리
  }, []);

  // ✅ 필터링 적용
  useEffect(() => {
    const lowerSearch = searchQuery.toLowerCase();
    const filtered = logs.filter((log) => {
      const matchSearch =
        log.message.toLowerCase().includes(lowerSearch) ||
        log.userId.toLowerCase().includes(lowerSearch) ||
        log.type.toLowerCase().includes(lowerSearch);

      const matchType =
        selectedType === "전체" || log.type === selectedType;

      return matchSearch && matchType;
    });

    setFilteredLogs(filtered);
  }, [logs, searchQuery, selectedType]);

  return (
    <main className="min-h-screen bg-gray-100 flex flex-col">
      <Navbar />

      <section className="flex-grow flex flex-col items-center p-8 gap-6">
        <h1 className="text-3xl font-bold text-gray-800">📋 전체 이벤트 로그</h1>

        {/* 🔍 검색 & 필터 */}
        <div className="w-full max-w-6xl flex flex-col md:flex-row justify-between items-center gap-4 mb-4">
          <input
            type="text"
            placeholder="검색어 입력 (유형, 사용자, 메시지)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full md:w-[60%] px-4 py-2 border border-gray-300 rounded-lg shadow-sm"
          />

          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg shadow-sm"
          >
            <option value="전체">전체</option>
            <option value="자리 침입">자리 침입</option>
            <option value="보안 탐지">보안 탐지</option>
            <option value="시스템 시작">시스템 시작</option>
            <option value="자리 복귀">자리 복귀</option>
          </select>
        </div>

        {/* ✅ 로그 목록 */}
        <div className="w-full max-w-6xl space-y-4">
          {filteredLogs.length === 0 ? (
            <p className="text-gray-600">일치하는 이벤트 로그가 없습니다.</p>
          ) : (
            filteredLogs.map((log, idx) => (
              <div
                key={idx}
                className="bg-white rounded-lg shadow p-4 border border-gray-200"
              >
                <div className="flex justify-between text-sm text-gray-500 mb-1">
                  <span>{log.time}</span>
                  <span className="font-semibold text-indigo-600">{log.type}</span>
                </div>
                <div className="text-gray-800">
                  👤 사용자 ID: <span className="font-medium">{log.userId}</span>
                </div>
                <div className="text-gray-700 mt-1">📌 {log.message}</div>
              </div>
            ))
          )}
        </div>
      </section>
    </main>
  );
}
