import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { query } from "../db/index.js";

export async function findByUsername(username) {
  const { rows } = await query(`SELECT * FROM users WHERE username=$1`, [
    username,
  ]);
  return rows[0] || null;
}

export async function createUser({ id, username, password, role = "viewer" }) {
  const hash = await bcrypt.hash(password, 10);
  await query(
    `INSERT INTO users (id, username, password_hash, role) VALUES ($1,$2,$3,$4)`,
    [id, username, hash, role]
  );
}

export async function verifyPassword(user, password) {
  return bcrypt.compare(password, user.password_hash);
}

export function signUserToken(user) {
  return jwt.sign(
    { sub: user.id, username: user.username, role: user.role },
    process.env.JWT_SECRET,
    { expiresIn: "7d" }
  );
}
