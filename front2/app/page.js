"use client";

import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-white">
      <div className="text-center bg-white p-8 rounded-lg">
        <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-black mb-6">
          OFFEYE: CCTV 보안 솔루션
          <br className="md:hidden" />
        </h1>
        <p className="text-gray-700 text-base md:text-lg leading-relaxed mb-8">
          실시간 블러처리와
          <br />
          자리침입 감지 기능을 제공합니다.
        </p>
        <Link href="/login">
          <button className="bg-black text-white px-6 py-2 rounded-lg hover:bg-gray-800 transition">
            시작하기
          </button>
        </Link>
      </div>
    </main>
  );
}
