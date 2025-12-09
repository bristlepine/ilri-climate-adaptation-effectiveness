'use client';

import { useState, useEffect } from "react";
import { ArrowRight } from "lucide-react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import insights from "@/lib/insights.json";

export default function InsightsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [filter, setFilter] = useState("All");
  const [yearFilter, setYearFilter] = useState("Year");
  const [visibleCount, setVisibleCount] = useState(9);
  const [scrolled, setScrolled] = useState(false);

  const categories = ["All", "Blog", "Webinar", "Article", "White Paper"];
  const years = [
    "Year",
    ...Array.from(new Set(insights.map((i) => i.year))).sort((a, b) => b - a),
  ];

  const handleClick = (item: (typeof insights)[0]) => {
    window.open(item.link, "_blank");
  };

  // Track scroll to toggle sticky blur
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50); // blur kicks in after 50px
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Filter + search
  const filteredInsights = insights.filter((item) => {
    const q = searchQuery.toLowerCase();
    const matchesSearch =
      item.title.toLowerCase().includes(q) ||
      item.authors.toLowerCase().includes(q);

    const matchesCategory = filter === "All" || item.type === filter;
    const matchesYear = yearFilter === "Year" || item.year === Number(yearFilter);

    return matchesSearch && matchesCategory && matchesYear;
  });

  const visibleInsights = filteredInsights.slice(0, visibleCount);

  return (
    <main className="page-wrapper">
      <Navbar />

      <section
        id="insights"
        className="relative min-h-[calc(80vh)] px-6 pt-32 pb-20 text-sand"
      >
        {/* Background Video */}
        <video
          autoPlay
          muted
          loop
          playsInline
          className="absolute inset-0 w-full h-full object-cover z-[-1]"
        >
          <source
            src="https://bristlepine.s3.us-east-2.amazonaws.com/hero1.webm"
            type="video/webm"
          />
        </video>

        {/* Dark overlay */}
        <div className="absolute inset-0 bg-black/60 z-0" />

        <div className="relative z-10 max-w-6xl mx-auto">
          {/* Sticky header + filters */}
          <div
            className={`sticky top-0 z-30 transition backdrop-blur-0 ${
              scrolled ? "backdrop-blur-md bg-transparent" : "bg-transparent"
            }`}
          >
            <div className="text-center pt-4 pb-6">
              <h1 className="text-4xl font-logo font-bold mb-6">
                Insights & Ideas
              </h1>
              <p className="text-lg font-tagline font-medium mb-8">
                Thoughts, research, and stories from the Bristlepine perspective.
              </p>

              {/* Search + Filters */}
              <div className="flex flex-col items-center gap-6">
                {/* Search */}
                <input
                  type="text"
                  placeholder="Search insights..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full md:w-1/2 px-4 py-2 rounded-full bg-white text-charcoal border border-gray-300 shadow-sm focus:outline-none focus:ring-2 focus:ring-green"
                />

                {/* Filter Buttons + Year Dropdown */}
                <div className="flex flex-wrap justify-center gap-3">
                  {categories.map((category) => (
                    <button
                      key={category}
                      onClick={() => setFilter(category)}
                      className={`px-4 py-1 rounded-full text-sm font-semibold transition ${
                        filter === category
                          ? "bg-green text-white"
                          : "bg-white text-charcoal border border-gray-300 hover:bg-green/10"
                      }`}
                    >
                      {category}
                    </button>
                  ))}

                  <select
                    value={yearFilter}
                    onChange={(e) => setYearFilter(e.target.value)}
                    className="px-4 py-1 rounded-full text-sm font-semibold bg-white text-charcoal border border-gray-300 hover:bg-green/10 focus:outline-none"
                  >
                    {years.map((y) => (
                      <option key={y} value={y}>
                        {y}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-10 px-4 mt-6">
            {visibleInsights.map((item, index) => (
              <div
                key={index}
                onClick={() => handleClick(item)}
                className="cursor-pointer group rounded-xl overflow-hidden bg-sand text-charcoal shadow-[0_2px_12px_rgba(202,194,181,0.25)] hover:shadow-[0_4px_16px_rgba(202,194,181,0.4)] transition-all duration-300 transform hover:scale-[1.015] flex flex-col"
              >
                <div className="relative h-48 overflow-hidden">
                  <img
                    src={item.image}
                    alt={item.title}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                  <div className="absolute inset-0 bg-black/40 group-hover:bg-black/50 transition-all duration-300 flex items-center justify-center">
                    <h3 className="text-lg font-bold text-sand text-center px-4">
                      {item.title}
                    </h3>
                  </div>
                </div>
                <div className="flex flex-col justify-between flex-1 p-4 text-left">
                  <p className="text-sm mb-2 italic">{item.authors}</p>
                  <div className="flex flex-wrap gap-2 mb-3">
                    <span className="px-3 py-1 bg-green text-white text-xs font-semibold rounded-full">
                      {item.type}
                    </span>
                    <span className="px-3 py-1 bg-gray-200 text-gray-800 text-xs font-semibold rounded-full">
                      {item.year}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-green group-hover:text-sage transition-colors duration-300">
                    <ArrowRight className="w-5 h-5" />
                    <span className="text-sm font-semibold">Read More</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Load More */}
          {visibleCount < filteredInsights.length && (
            <div className="text-center">
              <button
                onClick={() => setVisibleCount((prev) => prev + 9)}
                className="mt-12 px-6 py-2 rounded-full bg-green text-white hover:bg-sage transition"
              >
                Load More
              </button>
            </div>
          )}
        </div>
      </section>

      <Footer />
    </main>
  );
}
