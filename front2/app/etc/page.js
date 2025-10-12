"use client";

import Navbar from "../components/Navbar";

export default function EtcPage() {
  return (
    <main className="min-h-screen bg-gray-100 flex flex-col">
      <Navbar />

      <div className="flex flex-col items-center px-6 py-10 space-y-10">
        {/* ✅ 제목 */}
        <h1 className="text-3xl font-bold text-gray-800 text-center">
          프로젝트 개요 및 기능 설명
        </h1>

        {/* ✅ 이미지 1 + 설명 */}
        <div className="bg-white shadow-md rounded-xl p-6 w-full max-w-5xl">
          
          <h2 className="text-xl font-semibold text-gray-800 mb-2">기능 및 활용 분야</h2>
          <ul className="list-disc pl-6 text-gray-700 space-y-1 text-sm">
            <li>관리자용 웹 대시보드 형태</li>
            <li>개인정보 및 기업 내부 정보 탐지 및 블러링</li>
            <li>모니터, 노트북 화면 탐지 후 자동 블러 처리</li>
            <li>DeepSORT 기반 사람 추적 → 자리 침입 감지</li>
            <li>침입 감지 시 로그 기록 가능</li>
          </ul>
        </div>

        {/* ✅ 이미지 2 + 설명 */}
        <div className="bg-white shadow-md rounded-xl p-6 w-full max-w-5xl">
         
          <h2 className="text-xl font-semibold text-gray-800 mb-2">프로젝트 배경 및 목표</h2>
          <p className="text-gray-700 text-sm leading-relaxed">
            CCTV 영상에 개인의 모니터 화면이나 기업 내부 문서가 노출될 경우, 정보 접근 권한이 없는 사람이 물리적으로 접근해 민감한 정보를 확인할 수 있습니다.<br />
            이로 인해 <strong className="text-red-600">개인정보 또는 기업 내부 정보 유출</strong> 가능성이 발생할 수 있습니다.
          </p>
          <div className="mt-3 p-4 border border-blue-300 bg-blue-50 text-blue-900 text-sm rounded">
            <strong>주제:</strong> 개인정보 및 기업 내부정보 유출 방지를 위한 CCTV 보안 솔루션  
            <br />
            - YOLO 기반 보호 대상 정보 블러 처리  
            <br />
            - DeepSORT 기반 비인가 자리 접근 감지
          </div>
        </div>
      </div>
    </main>
  );
}
