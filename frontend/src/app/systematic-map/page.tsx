'use client';

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { useState } from "react";
import { Download } from "lucide-react";

const figures = [
  { file: '/map/temporal_trends.png',   title: 'Publications Over Time',     description: 'Number of included studies per publication year.' },
  { file: '/map/geographic_bar.png',    title: 'Geographic Distribution',    description: 'Top countries by study count.' },
  { file: '/map/producer_type_bar.png', title: 'Producer Types',             description: 'Breakdown of studies by agricultural producer type.' },
  { file: '/map/methodology_bar.png',   title: 'Methodological Approaches',  description: 'Primary methodological design across included studies.' },
  { file: '/map/domain_heatmap.png',    title: 'Process & Outcome Domains',  description: 'Adaptation process and outcome domains by producer type.' },
  { file: '/map/domain_type_bar.png',   title: 'Adaptation Domain Type',     description: 'Studies assessing processes, outcomes, or both.' },
  { file: '/map/equity_bar.png',        title: 'Equity & Inclusion',         description: 'Equity and inclusion dimensions addressed in studies.' },
];

export default function SystematicMapPage() {
  const [lightbox, setLightbox] = useState<string | null>(null);

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
          <p className="text-sm font-tagline uppercase tracking-widest text-clay mb-3">Deliverables D4 – D5</p>
          <h1 className="text-4xl md:text-5xl font-logo font-bold mb-4">Systematic Map</h1>
          <p className="text-lg font-tagline text-clay max-w-2xl">
            Evidence mapping across methods used to track climate adaptation processes and outcomes for smallholder producers in LMICs.
          </p>
        </div>
      </section>

      {/* Status Banner */}
      <div className="bg-yellow-50 border-b border-yellow-200 px-6 py-3">
        <p className="max-w-5xl mx-auto text-xs font-tagline text-yellow-800">
          <strong>Preliminary — Scopus corpus only.</strong> Multi-database integration (WoS, CAB Abstracts, AGRIS, Academic Search Premier) and full-text screening in progress. Final version expected May 2026.
        </p>
      </div>

      {/* ROSES Flow Diagram */}
      <section className="bg-white px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-start justify-between mb-6 gap-4">
            <div>
              <h2 className="text-2xl font-logo font-bold text-green">ROSES Flow Diagram</h2>
              <p className="font-tagline text-sm text-gray-500 mt-1">Record flow across all screening stages, following ROSES reporting standards.</p>
            </div>
            <a href="/map/roses_flow.png" download
              className="shrink-0 flex items-center gap-2 text-xs font-tagline font-semibold px-4 py-2 rounded-full bg-green text-white hover:bg-sage transition">
              <Download className="w-3.5 h-3.5" /> Download
            </a>
          </div>
          <div className="rounded-xl border border-gray-200 overflow-hidden bg-sand cursor-zoom-in hover:shadow-md transition-all duration-300"
            onClick={() => setLightbox('/map/roses_flow.png')}>
            <img src="/map/roses_flow.png" alt="ROSES Flow Diagram" className="w-full object-contain max-h-[600px]" />
          </div>
        </div>
      </section>

      {/* Searchable Database */}
      <section className="relative px-6 py-16 overflow-hidden">
        <video autoPlay muted loop playsInline className="absolute inset-0 w-full h-full object-cover z-[-1]">
          <source src="https://bristlepine.s3.us-east-2.amazonaws.com/2994205-uhd_3840_2160_30fps_clip.mp4" type="video/mp4" />
        </video>
        <div className="absolute inset-0 bg-black/70 z-0" />
        <div className="relative z-10 max-w-5xl mx-auto">
          <div className="flex items-start justify-between mb-6 gap-4">
            <div>
              <h2 className="text-2xl font-logo font-bold text-sand">Searchable Database</h2>
              <p className="font-tagline text-sm text-clay mt-1">All included studies with key metadata. Download the full CSV below.</p>
            </div>
            <a href="/map/step15_coded.csv" download
              className="shrink-0 flex items-center gap-2 text-xs font-tagline font-semibold px-4 py-2 rounded-full bg-white text-charcoal hover:bg-sand transition">
              <Download className="w-3.5 h-3.5" /> Download CSV
            </a>
          </div>
          {/* Placeholder */}
          <div className="rounded-xl border border-white/20 bg-white/10 backdrop-blur-sm p-12 text-center">
            <p className="font-logo text-xl text-sand/70 mb-2">Interactive Table</p>
            <p className="font-tagline text-sm text-clay">
              Full searchable, filterable database will appear here once data extraction is complete (D5.6, target May 2026).<br />
              Download the preliminary CSV above to explore included records now.
            </p>
          </div>
        </div>
      </section>

      {/* Evidence Map Figures */}
      <section className="bg-sand px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-2xl font-logo font-bold text-green mb-2">Evidence Map</h2>
          <p className="font-tagline text-sm text-gray-500 mb-10">Click any figure to enlarge.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {figures.map((fig) => (
              <div key={fig.file}
                className="group rounded-xl overflow-hidden bg-white shadow-[0_2px_12px_rgba(202,194,181,0.25)] hover:shadow-[0_4px_16px_rgba(202,194,181,0.4)] transition-all duration-300 transform hover:scale-[1.015] cursor-zoom-in flex flex-col"
                onClick={() => setLightbox(fig.file)}>
                <div className="relative overflow-hidden bg-white">
                  <img src={fig.file} alt={fig.title} className="w-full object-contain transition-transform duration-300 group-hover:scale-105" />
                </div>
                <div className="p-4 border-t border-gray-100">
                  <h3 className="font-logo font-bold text-charcoal text-base">{fig.title}</h3>
                  <p className="font-tagline text-xs text-gray-500 mt-1">{fig.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Lightbox */}
      {lightbox && (
        <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-6 cursor-zoom-out"
          onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="Enlarged figure" className="max-w-full max-h-full rounded-xl shadow-2xl" />
        </div>
      )}

      <Footer />
    </main>
  );
}
