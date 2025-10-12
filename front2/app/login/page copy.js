"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [keepSignedIn, setKeepSignedIn] = useState(false); // ✅ 상태 추가
  const [errorMsg, setErrorMsg] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();
    setErrorMsg("");

    try {
      const API_BASE = process.env.NEXT_PUBLIC_API_BASE;

      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // 서버가 원하면 remember 플래그도 같이 보냄(서버 미사용이어도 무해)
        body: JSON.stringify({ email, password, remember: keepSignedIn }),
      });

      const data = await res.json();

      if (res.ok) {
        // ✅ keep me signed in: 저장소 선택
        const storage = keepSignedIn ? localStorage : sessionStorage;
        const other    = keepSignedIn ? sessionStorage : localStorage;

        localStorage.setItem("access_token", data.access_token);
        if (data.user) localstorage.setItem("user", JSON.stringify(data.user));
        // 다른 저장소에 남아있을 수 있는 토큰/유저 제거
        other.removeItem("token");
        other.removeItem("user");

        router.push("/dashboard");
      } else {
        setErrorMsg(data.detail || data.message || "로그인 실패");
      }
    } catch (error) {
      console.error("로그인 중 오류 발생:", error);
      setErrorMsg("서버와 연결할 수 없습니다.");
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-white">
      <div className="w-full max-w-md px-8 py-6 bg-white rounded-xl">
        <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">
          관리자 로그인
        </h2>

        <form onSubmit={handleLogin}>
          <div className="mb-6">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="이메일을 입력하세요."
              className="w-full px-4 py-3 bg-white border border-gray-300 rounded-full"
              required
            />
          </div>

          <div className="mb-6">
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="비밀번호를 입력하세요."
              className="w-full px-4 py-3 bg-white border border-gray-300 rounded-full"
              required
            />
          </div>

          <div className="mb-6 flex items-center justify-between">
            <label className="flex items-center text-sm text-gray-700">
              <input
                type="checkbox"
                className="mr-2"
                checked={keepSignedIn}
                onChange={(e) => setKeepSignedIn(e.target.checked)} // ✅ 연동
              />
              Keep me signed in
            </label>
          </div>

          {errorMsg && (
            <p className="text-red-600 text-sm mb-4 text-center">{errorMsg}</p>
          )}

          <button
            type="submit"
            className="w-full py-3 rounded-full text-gray-800 font-semibold bg-white hover:bg-[#7BB94D] transition"
          >
            Log in
          </button>
        </form>

        {/* 🔽 회원가입 텍스트 링크 */}
        <div className="mt-6 text-center text-sm text-gray-600">
          계정이 없으신가요?{" "}
          <button
            type="button"
            onClick={() => router.push("/signup")}
            className="text-emerald-700 font-medium hover:underline"
          >
            회원가입
          </button>
        </div>
      </div>
    </div>
  );
}
