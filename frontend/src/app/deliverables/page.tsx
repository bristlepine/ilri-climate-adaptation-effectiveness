'use client';

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { deliverables, type DeliverableStatus } from "@/lib/deliverables";
import { ExternalLink } from "lucide-react";

const statusConfig: Record<DeliverableStatus, { label: string; classes: string }> = {
  submitted:     { label: 'Submitted',    classes: 'bg-green/10 text-green border-green/20' },
  'in-progress': { label: 'In Progress',  classes: 'bg-blue-50 text-blue-700 border-blue-200' },
  overdue:       { label: 'Overdue',      classes: 'bg-red-50 text-red-700 border-red-200' },
  pending:       { label: 'Pending',      classes: 'bg-yellow-50 text-yellow-700 border-yellow-200' },
  'not-started': { label: 'Not Started',  classes: 'bg-gray-100 text-gray-500 border-gray-200' },
};

const cardImages: Record<string, string> = {
  D1:  '/images/dashboard.jpg',
  D2:  '/images/adaptation.jpg',
  D3:  '/images/resilience.jpg',
  D4:  '/images/climate.jpg',
  D5:  '/images/pacific.jpg',
  D6:  '/images/morocco.jpg',
  D7:  '/images/urban.jpg',
  D8:  '/images/network.png',
  D9:  '/images/winwin.jpg',
  D10: '/images/pexels-tomfisk-3145153.jpg',
};

export default function DeliverablesPage() {
  return (
    <main className="page-wrapper min-h-screen flex flex-col">
      <Navbar />

      {/* Hero */}
      <section className="relative min-h-[40vh] flex items-end px-6 pt-32 pb-16 text-sand overflow-hidden">
        <video autoPlay muted loop playsInline className="absolute inset-0 w-full h-full object-cover z-[-1]">
          <source src="https://bristlepine.s3.us-east-2.amazonaws.com/2994205-uhd_3840_2160_30fps_clip.mp4" type="video/mp4" />
        </video>
        <div className="absolute inset-0 bg-black/65 z-0" />
        <div className="relative z-10 max-w-5xl mx-auto w-full">
          <p className="text-sm font-tagline uppercase tracking-widest text-clay mb-3">ILRI / CGIAR · 2025–2026</p>
          <h1 className="text-4xl md:text-5xl font-logo font-bold mb-4">Deliverables</h1>
          <p className="text-lg font-tagline text-clay max-w-2xl">
            All project outputs — protocols, systematic map, systematic review, and supporting materials — with links to Zenodo DOIs and GitHub releases.
          </p>
        </div>
      </section>

      {/* Cards Grid */}
      <section className="bg-sand px-6 py-16 flex-grow">
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {deliverables.map((d) => {
            const status = statusConfig[d.status];
            return (
              <div key={d.id}
                className="group rounded-xl overflow-hidden bg-white text-charcoal shadow-[0_2px_12px_rgba(202,194,181,0.25)] hover:shadow-[0_4px_16px_rgba(202,194,181,0.4)] transition-all duration-300 transform hover:scale-[1.015] flex flex-col">

                {/* Image header */}
                <div className="relative h-44 overflow-hidden">
                  <img
                    src={cardImages[d.id]}
                    alt={d.title}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                  <div className="absolute inset-0 bg-black/45 group-hover:bg-black/55 transition-all duration-300" />
                  {/* ID badge */}
                  <div className="absolute top-3 left-3 bg-white/90 text-charcoal font-logo font-bold text-sm px-3 py-1 rounded-full">
                    {d.id}
                  </div>
                  {/* Status badge */}
                  <div className={`absolute top-3 right-3 text-xs font-tagline font-semibold px-3 py-1 rounded-full border bg-white/90 ${status.classes}`}>
                    {status.label}
                  </div>
                  {/* Title overlay */}
                  <div className="absolute bottom-3 left-3 right-3">
                    <p className="text-xs font-tagline text-white/70 uppercase tracking-widest">{d.subtitle} · {d.due}</p>
                    <h3 className="text-lg font-logo font-bold text-white leading-tight">{d.title}</h3>
                  </div>
                </div>

                {/* Body */}
                <div className="flex flex-col flex-1 p-5">
                  <p className="text-xs font-tagline text-gray-400 uppercase tracking-widest mb-2">{d.type}</p>
                  <p className="font-tagline text-sm text-charcoal leading-relaxed flex-1">{d.description}</p>

                  {d.note && (
                    <p className="text-xs font-tagline text-yellow-700 bg-yellow-50 border border-yellow-200 px-3 py-2 rounded-lg mt-3">
                      {d.note}
                    </p>
                  )}

                  {/* Links */}
                  <div className="flex flex-wrap gap-2 mt-4">
                    {d.zenodo ? (
                      <a href={d.zenodo} target="_blank" rel="noopener noreferrer"
                        className="flex items-center gap-1.5 text-xs font-tagline font-semibold px-3 py-1.5 rounded-full bg-green text-white hover:bg-sage transition">
                        <ExternalLink className="w-3 h-3" /> Zenodo DOI
                      </a>
                    ) : (
                      <span className="flex items-center gap-1.5 text-xs font-tagline px-3 py-1.5 rounded-full bg-gray-100 text-gray-400 border border-gray-200">
                        DOI pending
                      </span>
                    )}
                    {d.github && (
                      <a href={d.github} target="_blank" rel="noopener noreferrer"
                        className="flex items-center gap-1.5 text-xs font-tagline font-semibold px-3 py-1.5 rounded-full bg-charcoal text-sand hover:bg-black transition">
                        <ExternalLink className="w-3 h-3" /> GitHub
                      </a>
                    )}
                    {d.cgspace && (
                      <a href={d.cgspace} target="_blank" rel="noopener noreferrer"
                        className="flex items-center gap-1.5 text-xs font-tagline font-semibold px-3 py-1.5 rounded-full bg-blue-600 text-white hover:bg-blue-700 transition">
                        <ExternalLink className="w-3 h-3" /> CGSpace
                      </a>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <Footer />
    </main>
  );
}
