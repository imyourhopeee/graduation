"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Navbar from "../components/Navbar";

export default function MyPage() {
  const router = useRouter();
  const [user, setUser] = useState(null);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE; 

  useEffect(() => {
    const token = localStorage.getItem("access_token") || sessionStorage.getItem("token");
    if (!token) {
      router.replace("/login");
      return;
    }

    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}/auth/me`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error("me failed");
        const me = await res.json();
        // 화면 표시는 권한을 무조건 '관리자'로 고정
        setUser({
          name: me.name || "사용자",
          email: me.email,
          role: "관리자",
          profileImg: "/user.png",
        });
      } catch {
        router.replace("/login");
      }
    };
    load();
  }, [API_BASE, router]);

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
