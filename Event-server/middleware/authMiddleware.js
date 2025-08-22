import jwt from "jsonwebtoken";

export function verifyAI(req, res, next) {
  const token = (req.headers.authorization || "").replace(/^Bearer\s+/i, "");
  if (!token) return res.status(401).json({ message: "no token" });
  try {
    const p = jwt.verify(token, process.env.AI_JWT_SECRET);
    if (p.role !== "ai") throw new Error("invalid role");
    req.ai = p; // 컨트롤러에서 p.camera_id 같은 정보 접근 가능
    next();
  } catch (e) {
    console.error("[verifyAI]", e.message);
    return res.status(401).json({ message: "invalid ai token" });
  }
}

export function requireUser(req, res, next) {
  const token = (req.headers.authorization || "").replace(/^Bearer\s+/i, "");
  if (!token) return res.status(401).json({ message: "no token" });
  try {
    const p = jwt.verify(token, process.env.JWT_SECRET);
    req.user = p;
    next();
  } catch (e) {
    console.error("[requireUser]", e.message);
    return res.status(401).json({ message: "invalid user token" });
  }
}
