'use client';

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { teamMembers } from "@/lib/data";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

const pccm = [
  { letter: 'P', label: 'Population', description: 'Smallholder producers and related agricultural production systems in LMICs — crop, livestock, fisheries/aquaculture, and agroforestry.' },
  { letter: 'C', label: 'Concept', description: 'Adaptation processes and outcomes — changes in capacities, practices, resilience, productivity, livelihoods, and risk reduction.' },
  { letter: 'C', label: 'Context', description: 'Climate hazards, climate-stress conditions, and agricultural settings in low- and middle-income countries (LMICs).' },
  { letter: 'M', label: 'Methodological Focus', description: 'Methods, tools, frameworks, indicators, and approaches used to track, evaluate, or quantify adaptation processes and outcomes.' },
];

const stages = [
  {
    number: '01',
    title: 'Systematic Map',
    description: 'Catalogues the full range of adaptation processes, outcomes, and measurement approaches used in the agriculture sector. Produces a searchable database and evidence gap map.',
    deliverables: 'D1 – D5',
    image: '/images/adaptation.jpg',
    href: '/systematic-map',
  },
  {
    number: '02',
    title: 'Systematic Review & Meta-Analysis',
    description: 'Focuses on methodological strengths, limitations, and suitability of measurement approaches for different users and contexts.',
    deliverables: 'D6 – D9',
    image: '/images/resilience.jpg',
    href: '/systematic-review',
  },
];

export default function AboutPage() {
  return (
    <main className="page-wrapper min-h-screen flex flex-col">
      <Navbar />

      {/* Hero */}
      <section className="relative min-h-[50vh] flex items-end px-6 pt-32 pb-16 text-sand overflow-hidden">
        <video autoPlay muted loop playsInline className="absolute inset-0 w-full h-full object-cover z-[-1]">
          <source src="https://bristlepine.s3.us-east-2.amazonaws.com/2994205-uhd_3840_2160_30fps_clip.mp4" type="video/mp4" />
        </video>
        <div className="absolute inset-0 bg-black/65 z-0" />
        <div className="relative z-10 max-w-4xl mx-auto">
          <p className="text-sm font-tagline uppercase tracking-widest text-clay mb-3">ILRI / CGIAR · 2025–2026</p>
          <h1 className="text-4xl md:text-5xl font-logo font-bold mb-4 leading-tight">Measuring What Matters</h1>
          <p className="text-lg font-tagline text-clay max-w-2xl">
            Tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector.
          </p>
        </div>
      </section>

      {/* Research Question */}
      <section className="bg-white px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-logo font-bold text-green mb-6">Primary Research Question</h2>
          <blockquote className="border-l-4 border-green pl-6 py-2 mb-6">
            <p className="text-xl font-logo text-charcoal italic leading-relaxed">
              "What adaptation processes and outcomes have been measured for smallholder producers in the agriculture sector in LMICs, and what methods have been used to track, evaluate, or quantify these processes and outcomes?"
            </p>
          </blockquote>
          <p className="font-tagline text-charcoal leading-relaxed">
            The project systematically identifies, characterises, and compares methods used to track and measure adaptation processes and outcomes targeting smallholder agricultural producers in LMICs — covering frameworks, indicators, M&amp;E systems, analytical approaches, participatory tools, and digital/data-driven methods.
          </p>
        </div>
      </section>

      {/* PCCM */}
      <section className="bg-sand px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-logo font-bold text-green mb-2">PCCM Framework</h2>
          <p className="font-tagline text-charcoal mb-8">The review is structured around a Population–Concept–Context–Methodological Focus framework.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {pccm.map((item) => (
              <div key={item.label} className="rounded-xl border border-gray-200 bg-white p-6 flex gap-4">
                <div className="w-10 h-10 rounded-full bg-green text-white flex items-center justify-center font-logo font-bold text-lg shrink-0">{item.letter}</div>
                <div>
                  <h3 className="font-logo font-bold text-charcoal text-lg mb-1">{item.label}</h3>
                  <p className="font-tagline text-sm text-gray-600">{item.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Two Stages — cards with images on dark background */}
      <section className="relative px-6 py-16 overflow-hidden">
        <video autoPlay muted loop playsInline className="absolute inset-0 w-full h-full object-cover z-[-1]">
          <source src="https://bristlepine.s3.us-east-2.amazonaws.com/2994205-uhd_3840_2160_30fps_clip.mp4" type="video/mp4" />
        </video>
        <div className="absolute inset-0 bg-black/70 z-0" />
        <div className="relative z-10 max-w-4xl mx-auto">
          <h2 className="text-2xl font-logo font-bold text-sand mb-2">Two-Stage Approach</h2>
          <p className="font-tagline text-clay mb-8">The project proceeds in two sequential stages.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {stages.map((stage) => (
              <Link key={stage.number} href={stage.href}
                className="group rounded-xl overflow-hidden bg-sand text-charcoal shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-[1.015] flex flex-col">
                <div className="relative h-48 overflow-hidden">
                  <img src={stage.image} alt={stage.title} className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105" />
                  <div className="absolute inset-0 bg-black/40 group-hover:bg-black/50 transition-all duration-300 flex items-center justify-center">
                    <span className="text-8xl font-logo font-bold text-white/60 drop-shadow-lg">{stage.number}</span>
                  </div>
                </div>
                <div className="p-6 flex flex-col flex-1">
                  <h3 className="font-logo font-bold text-charcoal text-xl mb-2">{stage.title}</h3>
                  <p className="font-tagline text-sm text-gray-600 leading-relaxed flex-1 mb-4">{stage.description}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-tagline font-semibold text-green bg-green/10 px-3 py-1 rounded-full">{stage.deliverables}</span>
                    <span className="flex items-center gap-1 text-xs font-tagline font-semibold text-green group-hover:gap-2 transition-all">
                      Explore <ArrowRight className="w-3.5 h-3.5" />
                    </span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="bg-white px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-logo font-bold text-green mb-2">The Team</h2>
          <p className="font-tagline text-charcoal mb-8">
            Conducted by <strong>Bristlepine Resilience Consultants</strong> on behalf of ILRI / CGIAR.
            Systematic review methodology support by <strong>Neal Haddaway</strong>.
            Project coordination by <strong>Aditi Mukherji</strong>, Principal Scientist, Climate Action, ILRI.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {teamMembers.map((member) => (
              <Link key={member.slug} href={`/team/${member.slug}`}
                className="group rounded-xl bg-sand shadow hover:shadow-md transition-all duration-300 transform hover:scale-[1.015] flex flex-col items-center text-center p-6">
                <div className="w-28 h-28 rounded-full overflow-hidden border-4 border-white shadow-md mb-4 shrink-0">
                  <img src={member.image} alt={member.name} className="w-full h-full object-cover object-top transition-transform duration-300 group-hover:scale-105" />
                </div>
                <h3 className="font-logo font-bold text-charcoal">{member.name}</h3>
                <p className="text-sm font-tagline text-gray-500 mb-3 flex-1">{member.shortBio}</p>
                <div className="flex items-center gap-2 text-green group-hover:text-sage transition-colors">
                  <ArrowRight className="w-4 h-4" />
                  <span className="text-sm font-semibold font-tagline">View Profile</span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
