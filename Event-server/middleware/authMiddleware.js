// middleware/authMiddleware.js
import jwt from "jsonwebtoken";

/**
 * AI → Event-server 간 서비스 토큰 검증
 * - Authorization: Bearer <AI 토큰(HS256, AI_JWT_SECRET)>
 * - payload.role === "ai" 여야 함
 */
// export function verifyAI(req, res, next) {
//   // A. 헤더 들어오는지 로깅 (문제 원인 파악용)
//   console.log("auth header:", req.headers.authorization);

//   const auth = req.headers.authorization || "";
//   if (!auth.startsWith("Bearer ")) {
//     return res.status(401).json({ message: "no token" });
//   }
//   const token = auth.slice("Bearer ".length).trim();

//   // B. 서버 설정 체크
//   const secret = process.env.AI_JWT_SECRET;
//   if (!secret) {
//     console.error("[verifyAI] AI_JWT_SECRET missing in env");
//     return res.status(500).json({ message: "server misconfigured" });
//   }

//   try {
//     // C. 실제 검증 (알고리즘 고정 + 약간의 clock skew 허용)
//     const payload = jwt.verify(token, secret, {
//       algorithms: ["HS256"],
//       clockTolerance: 30, // 시계 오차 허용(초)
//     });

//     // D. 필수 클레임
//     if (payload.role !== "ai") {
//       throw new Error("invalid role");
//     }

//     // camera_id, exp 등 접근 가능
//     req.ai = payload;
//     return next();
//   } catch (e) {
//     // 만료/서명/형식 등 상세 로그
//     console.error("[verifyAI] verify fail =>", {
//       name: e.name,
//       message: e.message,
//       now: Math.floor(Date.now() / 1000),
//     });

//     // 토큰 만료도 401로 응답 (클라이언트가 재발급하도록)
//     return res.status(401).json({ message: "invalid ai token" });
//   }
// }
export function verifyAI(req, res, next) {
  console.log("[verifyAI] headers:", req.headers);  // 🔍 추가 (AI 요청이 실제로 어떻게 오는지 보기)

   const h = req.headers.authorization || "";
   if (!h) {
     console.log("[verifyAI] no Authorization header at all");
     return res.status(401).json({ message: "no Authorization header" });
   }
   const m = h.match(/^Bearer\s+(.+)$/i);

   if (!m) {
     console.log("[verifyAI] malformed Authorization header:", h.slice(0, 20) + "...");
     return res.status(401).json({ message: "malformed bearer header" });
   }
   const token = m[1];

   if (!process.env.AI_JWT_SECRET) {
     console.error("[verifyAI] AI_JWT_SECRET missing");
     return res.status(500).json({ message: "server misconfigured: AI_JWT_SECRET" });
   }

   try {
     const payload = jwt.verify(token, process.env.AI_JWT_SECRET, {
       algorithms: ["HS256"],
       clockTolerance: 60,
     });

     console.log("[verifyAI] ok sub=%s role=%s exp=%s now=%s",
       payload?.sub, payload?.role, payload?.exp, Math.floor(Date.now()/1000));

     if (!payload?.role || payload.role.toLowerCase() !== "ai") {
      console.log("[verifyAI] role not ai:", payload?.role);
      return res.status(403).json({ message: "invalid ai token", reason: "role" });
    }

     req.ai = payload;
     return next();
   } catch (e) {
     console.error("[verifyAI] verify failed:", e.name, e.message);
     return res.status(401).json({ message: "invalid ai token", reason: e.name });
   }
}


/**
 * 프론트(사용자) → Event-server 호출용
 * - Authorization: Bearer <사용자 토큰(HS256, JWT_SECRET)>
 */
export function requireUser(req, res, next) {

  //추가1 - 게발 시 강제 우회 옵션 (없어도됨)
  if (process.env.DEV_BYPASS_AUTH === "1") {
    req.user = { sub: "dev", role: "admin", bypass: true };
    return next();
  }

  const auth = req.headers.authorization || "";
  if (!auth.startsWith("Bearer ")) {
    return res.status(401).json({ message: "no token" });
  }
  const token = auth.slice("Bearer ".length).trim();


  const secret = process.env.JWT_SECRET;
  if (!secret) {
    console.error("[requireUser] JWT_SECRET missing in env");
    return res.status(500).json({ message: "server misconfigured" });
  }

  try {
   
    const payload = jwt.verify(token, secret, {
      algorithms: ["HS256"],
      clockTolerance: 30,
    });

     console.log("[requireUser] ok sub=%s role=%s", payload?.sub ?? payload?.id, payload?.role);
     req.user = payload; // 예: sub, role 등
    return next();
  } catch (e) {
    console.error("[requireUser] verify fail =>", {
      name: e.name,
      message: e.message,
      now: Math.floor(Date.now() / 1000),
    });
    return res.status(401).json({ message: "invalid user token" });
  }
}