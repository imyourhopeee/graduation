// controllers/authController.js
import {
  findByUsername,
  createUser,
  verifyPassword,
  signUserToken,
} from "../models/userModel.js";

const isValidEmail = (v) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(v || "").trim());
const isValidPassword = (pw) => {
  if (typeof pw !== "string" || pw.length < 8) return false;
  return /[a-zA-Z]/.test(pw) && /\d/.test(pw) && /[^a-zA-Z0-9]/.test(pw);
};

// 회원가입: 프론트 { name, email, password } → 서버는 username 컬럼에 email 저장
export async function register(req, res) {
  try {
    const { name, password } = req.body || {};
    // ✅ email 또는 username 어느 쪽이 와도 받기
    const email = (req.body?.email ?? req.body?.username ?? "").trim();

    if (!email || !password) {
      return res.status(400).json({ message: "email, password는 필수입니다." });
    }
    if (!isValidEmail(email)) {
      return res.status(400).json({ message: "이메일 형식이 올바르지 않습니다." });
    }
    if (!isValidPassword(password)) {
      return res.status(400).json({
        message: "비밀번호는 8자 이상이며, 영문/숫자/특수문자를 모두 포함해야 합니다.",
      });
    }

    // DB는 username 컬럼 사용 → email 저장
    const user = await createUser({ username: email, password, name });
    return res.status(201).json({
      ok: true,
      user: { id: user.id, email: user.username, role: user.role, name: user.name },
    });
  } catch (e) {
    if (e?.status === 409 || e?.message === "USERNAME_TAKEN") {
      return res.status(409).json({ message: "이미 가입된 이메일입니다." });
    }
    console.error("[register] error", e);
    return res.status(500).json({ message: "서버 오류가 발생했습니다." });
  }
}

// 로그인: 프론트 { email, password } 또는 { username, password } 모두 허용
export async function login(req, res) {
  try {
    const password = req.body?.password;
    const emailOrUsername = (req.body?.email ?? req.body?.username ?? "").trim();

    if (!emailOrUsername || !password) {
      return res.status(400).json({ detail: "이메일/비밀번호를 입력하세요." });
    }

    const user = await findByUsername(emailOrUsername);
    if (!user) {
      return res.status(401).json({ detail: "이메일 또는 비밀번호가 올바르지 않습니다." });
    }

    const ok = await verifyPassword(user, password);
    if (!ok) {
      return res.status(401).json({ detail: "이메일 또는 비밀번호가 올바르지 않습니다." });
    }

    const token = signUserToken(user);
    return res.json({
      token,
      user: { id: user.id, email: user.username, role: user.role, name: user.name },
    });
  } catch (e) {
    console.error("[login] error", e);
    return res.status(500).json({ detail: "서버 오류가 발생했습니다." });
  }
}

//me 추가
export async function me(req, res) {
  try {
    const username = req.user?.username;
    if (!username) return res.status(400).json({ detail: "Bad token" });

    const user = await findByUsername(username);
    if (!user) return res.status(404).json({ detail: "Not found" });

    return res.json({
      id: user.id,
      email: user.username,
      name: user.name,
      role: user.role,
    });
  } catch {
    return res.status(500).json({ detail: "Server error" });
  }
}

