"use client";
import Navbar from "../components/Navbar";
import FilterableEventList from "../components/FilterableEventList";

export default function EventsPage() {
  return (
    <main className="min-h-screen bg-gray-100 flex flex-col">
      <Navbar />
      <section className="flex-grow p-8 max-w-5xl mx-auto w-full">
        <h1 className="text-3xl font-bold text-gray-800">📋 전체 이벤트 로그</h1>
        <div className="mt-4">
          <FilterableEventList limit={100} pollMs={5000} />
        </div>
      </section>
    </main>
  );
}
