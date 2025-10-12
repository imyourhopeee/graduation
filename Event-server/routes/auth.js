import { Router } from "express";
import jwt from "jsonwebtoken";
import { login, register, me} from "../controllers/authController.js";
import { authMiddleware } from "../middleware/auth.js";

const router = Router();
const DEMO_USER = { id: "u1", email: "jolpeuinpeuleon@gmail.com", name: "OFFEYE", password: "duksung1234!" };
router.post("/register", register);
router.post("/login", (req, res) => {
  const { email, password } = req.body || {};
  if (email !== DEMO_USER.email || password !== DEMO_USER.password) {
    return res.status(401).json({ message: "invalid credentials" });
  }
  const token = jwt.sign(
    { sub: DEMO_USER.id, email: DEMO_USER.email, name: DEMO_USER.name, role: "user" },
    process.env.JWT_SECRET,
    { algorithm: "HS256", expiresIn: "2h" }
  );
  res.json({ access_token: token });
});
router.get("/me", authMiddleware, me);

export default router;
