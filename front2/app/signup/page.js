"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function SignupPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [agree, setAgree] = useState(false);
  const [showPw, setShowPw] = useState(false);

  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE;

  const validate = () => {
    if (!name.trim()) return "이름을 입력해주세요.";
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))
      return "이메일 형식이 올바르지 않습니다.";
    if (password.length < 8) return "비밀번호는 8자 이상이어야 합니다.";

    const hasNum = /\d/.test(password);
    const hasLetter = /[a-zA-Z]/.test(password);
    const hasSpecial = /[^a-zA-Z0-9]/.test(password); // 특수문자 필수

    if (!hasNum || !hasLetter || !hasSpecial)
      return "비밀번호는 영문, 숫자, 특수문자를 모두 포함해야 합니다.";

    if (password !== confirm) return "비밀번호 확인이 일치하지 않습니다.";
    if (!agree) return "이용약관 및 개인정보 처리에 동의가 필요합니다.";
    return null;
  };

  const handleSignup = async (e) => {
    e.preventDefault();

    const v = validate();
    if (v) {
      alert(v); // 규칙 어긋나면 경고창
      return;
    }

    if (!API_BASE) {
      alert("서버 주소가 없습니다. NEXT_PUBLIC_API_BASE 를 설정하세요.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name.trim(),
          email: email.trim(),
          password,
        }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        const msg = data.message || data.detail || "회원가입에 실패했습니다.";
        throw new Error(msg);
      }

      setSuccess("회원가입이 완료되었습니다. 로그인 페이지로 이동합니다.");
      setTimeout(() => router.push("/login"), 800);
    } catch (err) {
      alert(err.message || "문제가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
      <div className="w-full max-w-md bg-white shadow-xl rounded-2xl p-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">회원가입</h1>
        <p className="text-sm text-gray-500 mb-6">
          OFFEYE 대시보드에 사용할 계정을 생성합니다.
        </p>

        <form className="space-y-4" onSubmit={handleSignup}>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              이름
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded-xl border border-gray-300 px-3 py-2 outline-none focus:ring-2 focus:ring-emerald-500"
              placeholder="홍길동"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              이메일
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-xl border border-gray-300 px-3 py-2 outline-none focus:ring-2 focus:ring-emerald-500"
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              비밀번호
            </label>
            <div className="relative">
              <input
                type={showPw ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-xl border border-gray-300 px-3 py-2 pr-10 outline-none focus:ring-2 focus:ring-emerald-500"
                placeholder="8자 이상, 영문+숫자+특수문자 포함"
              />
              <button
                type="button"
                onClick={() => setShowPw(!showPw)}
                className="absolute inset-y-0 right-0 px-3 text-sm text-gray-500 hover:text-gray-700"
              >
                {showPw ? "숨김" : "표시"}
              </button>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              비밀번호 확인
            </label>
            <input
              type={showPw ? "text" : "password"}
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              className="w-full rounded-xl border border-gray-300 px-3 py-2 outline-none focus:ring-2 focus:ring-emerald-500"
              placeholder="비밀번호를 다시 입력"
            />
          </div>

          <label className="flex items-start gap-2 text-sm text-gray-600 select-none">
            <input
              type="checkbox"
              checked={agree}
              onChange={(e) => setAgree(e.target.checked)}
              className="mt-1 h-4 w-4 rounded border-gray-300 text-emerald-600 focus:ring-emerald-500"
            />
            <span>
              <span className="font-medium">이용약관</span> 및{" "}
              <span className="font-medium">개인정보 처리방침</span>에
              동의합니다.
            </span>
          </label>

          {success && (
            <div className="rounded-xl bg-emerald-50 text-emerald-700 text-sm p-3">
              {success}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-xl bg-emerald-600 text-white py-2.5 font-medium shadow hover:bg-emerald-700 focus:outline-none focus:ring-4 focus:ring-emerald-200 disabled:opacity-60"
          >
            {loading ? "가입 중…" : "회원가입"}
          </button>
        </form>

        <div className="mt-6 text-center text-sm text-gray-600">
          이미 계정이 있으신가요?{" "}
          <button
            type="button"
            onClick={() => router.push("/login")}
            className="text-emerald-700 font-medium hover:underline"
          >
            로그인
          </button>
        </div>
      </div>
    </main>
  );
}
