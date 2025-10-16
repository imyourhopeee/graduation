"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Navbar from "../components/Navbar";

export default function MyPage() {
  const router = useRouter();
  const [user, setUser] = useState(null);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE; 

useEffect(() => {
  // 1) API_BASE 안전하게 만들기
  const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE ||
    process.env.NEXT_PUBLIC_EVENT_URL ||
    "http://localhost:3002";
  const base = API_BASE.replace(/\/+$/, "");

  // 2) 토큰은 로그인 때 저장한 'access_token'만 사용 (헤더 방식 고정)
  const token = localStorage.getItem("access_token") || sessionStorage.getItem("access_token");
  if (!token) {
    console.warn("[mypage] no access_token in storage");
    router.replace("/login");
    return;
  }

  // 3) /auth/me 경로 호출 (만약 서버에서 /api로 묶었으면 '/api/auth/me'로 바꾸세요!)
  const meUrl = `${base}/auth/me`;

  (async () => {
    try {
      const res = await fetch(meUrl, {
        headers: { Authorization: `Bearer ${token}` },
      });

      // 디버깅 로그 (필요 없으면 나중에 지워도 됨)
      console.debug("[mypage] GET", meUrl, "→", res.status);

      if (!res.ok) throw new Error(`me failed: ${res.status}`);
      const me = await res.json();

      setUser({
        name: me?.name || "사용자",
        email: me?.email || "",
        role: "관리자",
        profileImg: "/user.png",
      });
    } catch (e) {
      console.warn("[mypage] /auth/me 실패:", e?.message);
      router.replace("/login");
    }
  })();
}, [router]);



  if (!user) {
    return (
      <main className="min-h-screen bg-gray-100 flex flex-col">
        <Navbar />
        <div className="flex flex-col items-center py-12 px-4 flex-grow">
          <div className="w-full max-w-md p-8 bg-white rounded-xl shadow">
            <div className="animate-pulse space-y-4">
              <div className="w-32 h-32 mx-auto rounded-full bg-gray-200" />
              <div className="h-4 bg-gray-200 rounded w-1/2 mx-auto" />
              <div className="space-y-2">
                <div className="h-3 bg-gray-200 rounded" />
                <div className="h-3 bg-gray-200 rounded w-4/5" />
              </div>
            </div>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-gray-100 flex flex-col">
      <Navbar />
      <div className="flex flex-col items-center py-12 px-4 flex-grow">
        <div className="flex flex-col items-center space-y-4 bg-white rounded-xl shadow-lg p-8 w-full max-w-md">
          <img
            src={user.profileImg || "/user.png"}
            alt="프로필 이미지"
            className="w-32 h-32 rounded-full object-cover border-4 border-green-400"
          />
          <h2 className="text-2xl font-bold text-gray-800">{user.name}</h2>
          <div className="w-full space-y-2 text-sm text-gray-700">
            <div className="flex justify-between">
              <span className="font-medium text-gray-600">이메일</span>
              <span>{user.email}</span>
            </div>
            <div className="flex justify-between">
              <span className="font-medium text-gray-600">권한</span>
              <span>{user.role}</span> {/* '관리자'로 고정 */}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
