//login
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import crypto from "node:crypto";
import { query } from "../db/index.js";

const ROUNDS = Number(process.env.BCRYPT_ROUNDS || 10);

export async function findByUsername(username) {
  const u = username.trim();
  const { rows } = await query(`SELECT * FROM users WHERE username=$1`, [u]);
  return rows[0] || null;
}

export async function createUser({ id, username, password, role = "viewer", name = null }) {
  const uid = id || crypto.randomUUID();              // 1) id 자동 생성
  const u = username.trim();                          // 3) 정규화
  const hash = await bcrypt.hash(password, ROUNDS);

  try {
    await query(
      `INSERT INTO users (id, username, password_hash, role, name)
       VALUES ($1,$2,$3,$4,$5)`,
      [uid, u, hash, role, name]
    );
    return { id: uid, username: u, role, name };
  } catch (e) {
    if (e.code === "23505") {                         // 2) 중복 처리
      const err = new Error("USERNAME_TAKEN");
      err.status = 409;
      throw err;
    }
    throw e;
  }
}

export async function verifyPassword(user, password) {
  return bcrypt.compare(password, user.password_hash);
}

export function signUserToken(user) {
  if (!process.env.JWT_SECRET) {
    throw new Error("JWT_SECRET is not set");
  }
  return jwt.sign(
    { sub: user.id, username: user.username, role: user.role },
    process.env.JWT_SECRET,
    { expiresIn: "7d" }
  );
}
