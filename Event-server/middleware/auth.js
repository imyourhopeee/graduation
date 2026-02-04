import jwt from "jsonwebtoken";

export function authMiddleware(req, res, next) {
  try {
    const header = req.headers.authorization || "";
    const token = header.startsWith("Bearer ") ? header.slice(7) : null;
    if (!token) return res.status(401).json({ detail: "Unauthorized" });
    const payload = jwt.verify(token, process.env.JWT_SECRET);
    req.user = payload; // { sub, username, role, ... }
    next();
  } catch {
    return res.status(401).json({ detail: "Unauthorized" });
  }
}
