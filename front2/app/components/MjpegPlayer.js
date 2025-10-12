"use client";

export default function MjpegPlayer({ src, style }) {
  return <img src={src} style={{ width: "100%", ...style }} alt="stream" />;
}
