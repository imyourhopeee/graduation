// middleware/authMiddleware.js
import jwt from "jsonwebtoken";

export function verifyAI(req, res, next) {
  console.log('auth header:', req.headers.authorization);
  const token = (req.headers.authorization || "").replace(/^Bearer\s+/i, "");
  
  if (!token) return res.status(401).json({ message: "no token" });

  try {
    const decoded = jwt.verify(token, process.env.AI_JWT_SECRET);
    if (decoded.role !== "ai") throw new Error("invalid role");

    req.ai = decoded; // camera_id, exp 등 접근 가능
    next();
  } catch (e) {
    console.error("[verifyAI]", e.message);
    return res.status(401).json({ message: "invalid ai token" });
  }
}

// export function verifyAI(req, res, next) {
//   const auth = req.headers.authorization || "";
//   const token = auth.replace(/^Bearer\s+/i, "");
//   if (!token) {
//     console.error("[verifyAI] no token. headers=", req.headers);
//     return res.status(401).json({ message: "no token" });
//   }

//   try {
//     const secret = process.env.AI_JWT_SECRET;
//     if (!secret) {
//       console.error("[verifyAI] AI_JWT_SECRET missing in env");
//       return res.status(500).json({ message: "server misconfigured" });
//     }

//     // (A) 일단 원인 파악용: 디코드 결과를 한번 찍는다
//     const decodedLoose = jwt.decode(token, { complete: true });
//     console.log("[verifyAI] decoded (no verify):", decodedLoose);

//     // (B) 실제 검증
//     const payload = jwt.verify(token, secret /*, { algorithms: ["HS256"] }*/);

//     // role 필수
//     if (payload.role !== "ai") throw new Error("invalid role");

//     req.ai = payload;
//     return next();

//   } catch (e) {
//     console.error("[verifyAI] verify fail =>", {
//       name: e.name, message: e.message,
//       // 만료 문제인지 바로 보자
//       now: Math.floor(Date.now()/1000),
//       secret_len: (process.env.AI_JWT_SECRET || "").length,
//       auth_sample: (req.headers.authorization || "").slice(0, 40) + "...",
//     });

//     // 만약 만료 의심이면 아래처럼 테스트:
//     // try {
//     //   const payloadNoExp = jwt.verify(token, process.env.AI_JWT_SECRET, { ignoreExpiration: true });
//     //   console.error("[verifyAI] would pass if ignoreExpiration. payload=", payloadNoExp);
//     // } catch {}

//     return res.status(401).json({ message: "invalid ai token" });
//   }
// }
// 프론트(사용자) → Event-server 호출용
export function requireUser(req, res, next) {
  const token = (req.headers.authorization || "").replace(/^Bearer\s+/i, "");
  if (!token) return res.status(401).json({ message: "no token" });
  try {
    const p = jwt.verify(token, process.env.JWT_SECRET);
    req.user = p; // 예: p.sub, p.role 등
    next();
  } catch (e) {
    console.error("[requireUser]", e.message);
    return res.status(401).json({ message: "invalid user token" });
  }
}
