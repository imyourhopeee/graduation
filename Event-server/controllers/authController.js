import {
  createUser,
  findByUsername,
  verifyPassword,
  signUserToken,
} from "../models/userModel.js";
import { randomUUID } from "crypto";

export async function register(req, res, next) {
  try {
    const { username, password, role } = req.body;
    const exist = await findByUsername(username);
    if (exist) return res.status(409).json({ message: "username exists" });
    await createUser({
      id: randomUUID(),
      username,
      password,
      role: role || "viewer",
    });
    res.status(201).json({ ok: true });
  } catch (e) {
    next(e);
  }
}

export async function login(req, res, next) {
  try {
    const { username, password } = req.body;
    const user = await findByUsername(username);
    if (!user) return res.status(401).json({ message: "invalid credentials" });
    const ok = await verifyPassword(user, password);
    if (!ok) return res.status(401).json({ message: "invalid credentials" });
    const token = signUserToken(user);
    res.json({
      token,
      user: { id: user.id, username: user.username, role: user.role },
    });
  } catch (e) {
    next(e);
  }
}
