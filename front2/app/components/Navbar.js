// components/Navbar.js
"use client";

import Link from "next/link";

export default function Navbar() {
  const menuItems = [
    { name: "Dashboard", href: "/dashboard" },
    { name: "Events", href: "/events" },
    { name: "Settings", href: "/settings" },
    { name: "Seats", href: "/seats" },
    { name: "Mypage", href: "/mypage" },
    { name: "Etc", href: "/etc" },
  ];

  return (
    <header className="bg-black shadow p-4 flex justify-between items-center">
      <Link href="/mypage">
        <img
          src="/user.png"
          alt="User Profile"
          className="w-10 h-10 rounded-full object-cover cursor-pointer"
        />
      </Link>

      <nav className="flex space-x-4">
        {menuItems.map((item) => (
          <Link
            key={item.name}
            href={item.href}
            className={`text-sm px-3 py-2 rounded cursor-pointer hover:bg-green ${
              item.name === "Anonymization"
                ? "font-semibold text-indigo-600"
                : "text-white"
            }`}
          >
            {item.name}
          </Link>
        ))}
      </nav>
    </header>
  );
}