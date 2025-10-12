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
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ email, password, remember: keepSignedIn }),
      credentials: "include", // 쿠키도 쓰는 서버면 유용, 아니면 있어도 무해
    });

    // 응답 본문 안전 파싱
    const raw = await res.text();
    let data;
    try { data = raw ? JSON.parse(raw) : {}; } catch { data = {}; }

    if (!res.ok) {
      setErrorMsg(data.detail || data.message || `로그인 실패 (${res.status})`);
      return;
    }

    // 서버가 주는 토큰 키 다양한 케이스 대응
    const token =
      data.access_token ??
      data.token ??
      data.jwt ??
      data.accessToken ??
      data?.data?.access_token ??
      null;

    // keep me signed in에 따라 저장소 선택
    const storage = keepSignedIn ? localStorage : sessionStorage;
    const other   = keepSignedIn ? sessionStorage : localStorage;

    if (token) {
      storage.setItem("access_token", token);          // ✅ 핵심: 통일된 키 이름
    } else {
      // httpOnly 쿠키만 쓰는 서버일 수도 있음 → 대시보드가 쿠키모드(fetch credentials: 'include')여야 함
      // 지금은 대시보드가 Bearer 토큰을 요구하므로, 토큰이 없으면 안내
      setErrorMsg("로그인 성공했지만 토큰이 없습니다. 서버 응답을 확인하세요.");
      return;
    }

    // 유저 정보도 오면 저장(선택)
    if (data.user) {
      try { storage.setItem("user", JSON.stringify(data.user)); } catch {}
    }

    // 반대 저장소에 남아있을 수 있는 잔여 데이터 정리
    other.removeItem("access_token");
    other.removeItem("user");

    router.push("/dashboard");
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
