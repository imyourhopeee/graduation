import jwt from "jsonwebtoken";

const secret = "2ba4e04dbeeb45459614a86b132a775e"; // Event-server의 AI_JWT_SECRET과 동일
const token = jwt.sign({ role: "ai", name: "ai-server" }, secret, { expiresIn: "30d" });

console.log(token);
