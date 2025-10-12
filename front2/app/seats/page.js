"use client";

import { useEffect, useRef, useState } from "react";
import MjpegPlayer from "../components/MjpegPlayer";
import Navbar from "../components/Navbar";

const ai = (process.env.NEXT_PUBLIC_AI_URL || "http://localhost:3001").replace(/\/$/, "");
const STREAM_URL = `${ai}/stream/blur?cam=0&blur=0&intrusion=0&roi=1&w=1280&h=960&quality=85`;
const API_URL = `${ai}/config/seats`;


const vAdd = (a,b)=>[a[0]+b[0], a[1]+b[1]];
const vSub = (a,b)=>[a[0]-b[0], a[1]-b[1]];
const vMul = (a,k)=>[a[0]*k, a[1]*k];
const vLen = (a)=>Math.hypot(a[0], a[1]);


function drawSeat(ctx, seat) {
  const { p1, p2, inward_sign, d_near, d_far } = seat;

  // 노란 좌석선
  ctx.strokeStyle = "rgb(255,255,0)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(p1[0], p1[1]);
  ctx.lineTo(p2[0], p2[1]);
  ctx.stroke();

  // 수직 벡터 n
  const ax = p1[0], ay = p1[1], bx = p2[0], by = p2[1];
  const vx = bx - ax, vy = by - ay;
  const vlen = Math.hypot(vx, vy);
  if (vlen < 1e-3) return;

  const y_near = Math.max(p1[1], p2[1]);
  const y_far = Math.min(p1[1], p2[1]);

  const n_raw = [-vy, vx];
  const nlen = Math.hypot(n_raw[0], n_raw[1]) + 1e-6;
  const n = [(n_raw[0] / nlen) * inward_sign, (n_raw[1] / nlen) * inward_sign];

  const depthAtY = (fy) => {
    if (y_near === y_far) return d_near;
    const fyc = Math.max(Math.min(fy, y_near), y_far);
    const t = (fyc - y_far) / Math.max(1, (y_near - y_far));
    return d_far * (1 - t) + d_near * t;
  };

  const d1 = depthAtY(p1[1]);
  const d2 = depthAtY(p2[1]);

  const a2 = [ax + n[0] * d1, ay + n[1] * d1];
  const b2 = [bx + n[0] * d2, by + n[1] * d2];

  // 보라 사다리꼴
  ctx.fillStyle = "rgba(200,0,200,0.25)";
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.lineTo(b2[0], b2[1]);
  ctx.lineTo(a2[0], a2[1]);
  ctx.closePath();
  ctx.fill();

  // 안쪽 화살표
  const mid = [(ax + bx) / 2, (ay + by) / 2];
  const tip = [mid[0] + n[0] * 40, mid[1] + n[1] * 40];
  ctx.strokeStyle = "rgb(255,0,0)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(mid[0], mid[1]);
  ctx.lineTo(tip[0], tip[1]);
  ctx.stroke();
}

export default function SeatsPage() {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [videoSize, setVideoSize] = useState({ w: 0, h: 0 });

  const [seats, setSeats] = useState([]);
  const [tempPts, setTempPts] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);

  // 좌석 불러오기
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(API_URL);
        if (!res.ok) { setSeats([]); return; }
        const data = await res.json();
        setSeats(Array.isArray(data) ? data : []);
      } catch {
        setSeats([]);
      }
    })();
  }, []);

  // 컨테이너 크기 추적
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const cr = entry.contentRect;
        setVideoSize({ w: Math.round(cr.width), h: Math.round(cr.height) });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // 캔버스 그리기 루프
  useEffect(() => {
    let raf;
    const loop = () => {
      const canvas = canvasRef.current;
      if (!canvas || !videoSize.w || !videoSize.h) {
        raf = requestAnimationFrame(loop);
        return;
      }
      const ctx = canvas.getContext("2d");
      canvas.width = videoSize.w;
      canvas.height = videoSize.h;
      canvas.style.width = `${videoSize.w}px`;
      canvas.style.height = `${videoSize.h}px`;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      seats.forEach((seat, i) => {
        drawSeat(ctx, seat);
        ctx.fillStyle = i === selected ? "rgb(0,200,255)" : "white";
        ctx.font = "14px sans-serif";
        const label = `Seat ${Number.isFinite(seat?.seat_id) ? seat.seat_id : i}`;
        ctx.fillText(label, seat.p1[0], seat.p1[1] - 6);
      });

      if (tempPts.length === 1) {
        const [p] = tempPts;
        ctx.fillStyle = "rgb(0,200,255)";
        ctx.beginPath();
        ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
        ctx.fill();
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [videoSize, seats, tempPts, selected]);

  // 클릭 → 렌더 크기 좌표로 저장
  const onCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || !videoSize.w || !videoSize.h) return;
    const rect = canvas.getBoundingClientRect();
    const sx = videoSize.w / rect.width;
    const sy = videoSize.h / rect.height;
    const x = Math.round((e.clientX - rect.left) * sx);
    const y = Math.round((e.clientY - rect.top) * sy);
    const p = [x, y];

    if (tempPts.length === 0) {
      setTempPts([p]);
    } else {
      const p1 = tempPts[0];
      const p2 = p;
      const idx = seats.length;
      const newSeat = {
        p1, p2,
        inward_sign: 1,
        d_near: 180.0, d_far: 120.0,
        seat_id: idx,
      };
      setSeats((prev) => [...prev, newSeat]);
      setTempPts([]);
      setSelected(idx);
    }
  };

  const toggleInward = () => {
    if (selected == null) return;
    setSeats((prev) =>
      prev.map((s, i) => i === selected ? { ...s, inward_sign: (s.inward_sign >= 0 ? -1 : 1) } : s)
    );
  };

  const setDepth = (key, val) => {
    if (selected == null) return;
    setSeats((prev) =>
      prev.map((s, i) => (i === selected ? { ...s, [key]: Number(val) } : s))
    );
  };

  const deleteLast = () => {
    if (!seats.length) return;
    const arr = seats.slice(0, -1);
    setSeats(arr);
    setSelected(arr.length ? arr.length - 1 : null);
  };
  


  const saveSeats = async () => {
    setLoading(true);
    try {
      // seats가 배열이 아닐 수도 있으니 방어
      const norm = (Array.isArray(seats) ? seats : []).map((seat) => ({
        p1: [Math.round(Number(seat?.p1?.[0] ?? 0)), Math.round(Number(seat?.p1?.[1] ?? 0))],
        p2: [Math.round(Number(seat?.p2?.[0] ?? 0)), Math.round(Number(seat?.p2?.[1] ?? 0))],
        d_near: Number(seat?.d_near ?? 0),
        d_far: Number(seat?.d_far ?? 0),
        inward_sign: Number(seat?.inward_sign ?? 1) >= 0 ? 1 : -1,
        seat_id: Number(seat?.seat_id ?? 0),
      }));

      const res = await fetch(`${ai}/config/seats`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(norm),
      });

      const text = await res.text();
      if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} :: ${text}`);

      // 저장 직후 엔진 메모리 확인(스트림 HUD seats:N 갱신용)
      if (typeof reloadSeats === "function") await reloadSeats();
      alert("좌석 구성이 저장되었습니다.");
    } catch (e) {
      alert("저장 실패: " + (e?.message ?? e));
    } finally {
      setLoading(false);
    }
  };

  const reloadSeats = async () => {
    setLoading(true);
    try {
      const res = await fetch(API_URL);
      const data = await res.json();
      setSeats(Array.isArray(data) ? data : []);
      setSelected(null);
      setTempPts([]);
    } catch (e) {
      alert("불러오기 실패: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-100 flex flex-col">
      {/* 상단 네비 동일 */}
      <Navbar />

      {/* dashboard와 동일한 섹션 패딩/정렬 */}
      <section className="flex-grow flex flex-col items-center justify-start p-10 gap-6">

        {/* 1) 실시간 영상 카드 (동일 폭/디자인, 높이 70vh 고정) */}
        <div className="w-full max-w-6xl bg-white rounded-xl shadow-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-3xl font-bold text-gray-800">좌석(ROI) 설정</h1>
            <div className="flex gap-2">
              <button
                onClick={reloadSeats}
                className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300"
              >
                불러오기
              </button>
              <button
                onClick={saveSeats}
                disabled={loading}
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
              >
                {loading ? "저장 중..." : "저장"}
              </button>
            </div>
          </div>

          {/* 영상 + 캔버스: dashboard와 동일하게 중앙 정렬, 70vh */}
          <div className="flex justify-center">
            <div
              className="relative w-full"
              style={{ height: "70vh" }}
              ref={containerRef}
            >
              {/* 영상 (아래 캔버스가 클릭을 받도록 pointer-events 차단) */}
              <div className="absolute inset-0 pointer-events-none">
                <MjpegPlayer src={STREAM_URL} style={{ width: "100%", height: "100%" }} />
              </div>

              {/* 클릭 가능한 오버레이 캔버스 */}
              <canvas
                ref={canvasRef}
                onClick={onCanvasClick}
                className="absolute inset-0 w-full h-full z-10 pointer-events-auto cursor-crosshair"
              />
            </div>
          </div>
        </div>

        {/* 2) 편집 패널 카드 (dashboard의 이벤트 로그 카드 스타일과 통일) */}
        <div className="w-full max-w-6xl bg-white rounded-xl shadow p-4 border border-gray-300">
          <h2 className="text-lg font-semibold mb-3 text-gray-700">🎛️ 좌석 편집</h2>

          {selected != null && seats[selected] ? (
            <div className="space-y-3">
              <div className="text-sm text-gray-700">
                선택 좌석: <span className="font-semibold">{seats[selected].seat_id ?? selected}</span>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <button
                  onClick={toggleInward}
                  className="px-3 py-2 bg-gray-800 text-white rounded hover:bg-gray-900"
                >
                  안쪽 방향 토글 (inward_sign: {seats[selected].inward_sign})
                </button>

                <div className="min-w-[240px]">
                  <label className="block text-sm text-gray-600 mb-1">
                    d_near (아래쪽 깊이): {seats[selected].d_near}px
                  </label>
                  <input
                    type="range"
                    min="20"
                    max="400"
                    step="5"
                    value={seats[selected].d_near}
                    onChange={(e) => setDepth("d_near", e.target.value)}
                    className="w-full"
                  />
                </div>

                <div className="min-w-[240px]">
                  <label className="block text-sm text-gray-600 mb-1">
                    d_far (위쪽 깊이): {seats[selected].d_far}px
                  </label>
                  <input
                    type="range"
                    min="20"
                    max="400"
                    step="5"
                    value={seats[selected].d_far}
                    onChange={(e) => setDepth("d_far", e.target.value)}
                    className="w-full"
                  />
                </div>

                <button
                  onClick={deleteLast}
                  className="px-3 py-2 bg-red-500 text-white rounded hover:bg-red-600"
                >
                  마지막 좌석 삭제
                </button>
              </div>

              <p className="text-sm text-gray-500">
                캔버스를 클릭해서 좌석선을 추가하세요. (두 점 클릭)
              </p>
            </div>
          ) : (
            <p className="text-sm text-gray-500">
              편집할 좌석을 추가하거나 선택하세요.
            </p>
          )}
        </div>
      </section>
    </main>
  );
}
