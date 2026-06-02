'use client';

import { useEffect, useRef, useState } from "react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import PlotlyChart from "@/components/PlotlyChart";
import { ExternalLink } from "lucide-react";

// ─── Data ────────────────────────────────────────────────────────────────────

const pipelineStats = [
  { value: '40,634', label: 'Records identified (29 sources)' },
  { value: '26,173', label: 'Unique after deduplication' },
  { value: '8,753',  label: 'Included at T&A screening' },
  { value: '2,750',  label: 'Included at full-text screening' },
];

const calibrationRounds = [
  { round: 'R1 — initial criteria',    n: 205, sens: '0.776', spec: '0.703', prec: '0.559', f1: '0.650', kappa: '0.436', humanK: '0.500', pass: false },
  { round: 'R1a — 1st revision',       n: 205, sens: '0.761', spec: '0.797', prec: '0.646', f1: '0.699', kappa: '0.534', humanK: '0.500', pass: false },
  { round: 'R1b — 2nd revision',       n: 205, sens: '0.866', spec: '0.819', prec: '0.699', f1: '0.773', kappa: '0.645', humanK: '0.500', pass: false },
  { round: 'R2a — 3rd revision',       n: 103, sens: '0.897', spec: '0.905', prec: '0.788', f1: '0.839', kappa: '0.770', humanK: '0.765', pass: false },
  { round: 'R2b — 4th revision',       n: 103, sens: '0.966', spec: '0.838', prec: '0.700', f1: '0.812', kappa: '0.720', humanK: '0.765', pass: true  },
  { round: 'R3a — stability check',    n: 107, sens: '0.970', spec: '0.824', prec: '0.711', f1: '0.821', kappa: '0.721', humanK: '0.703', pass: true  },
];

const benchmarks = [
  { label: 'O\'Mara-Eves target',            sens: '≥ 0.95',  spec: '—',     prec: '—',     f1: '—',     kappa: '≥ 0.60', humanK: '—' },
  { label: 'Human screeners (Hanegraaf 2024)', sens: '—',      spec: '—',     prec: '—',     f1: '—',     kappa: '—',      humanK: '0.82 (abs) / 0.77 (FT)' },
  { label: 'AI — GPT-4 (Zhan 2025)',          sens: '0.992',  spec: '0.836', prec: '—',     f1: '—',     kappa: '0.83',   humanK: '—' },
  { label: 'AI mean, 172 studies (Scherbakov 2025)', sens: '0.804', spec: '—', prec: '0.632', f1: '0.708', kappa: '—',    humanK: '—' },
];



const softwareStack = [
  { component: 'LLM inference',        tool: 'Ollama (qwen2.5:14b)',            purpose: 'Local deterministic screening and data extraction' },
  { component: 'Scopus API',           tool: 'Elsevier REST API',               purpose: 'Record retrieval and abstract enrichment' },
  { component: 'DOI enrichment',       tool: 'Crossref, OpenAlex, Semantic Scholar', purpose: 'DOI lookup and abstract retrieval' },
  { component: 'Open access',          tool: 'Unpaywall API',                   purpose: 'Full-text URL discovery' },
  { component: 'PDF parsing',          tool: 'pypdf',                           purpose: 'Full-text extraction from PDFs' },
  { component: 'HTML parsing',         tool: 'trafilatura, BeautifulSoup4',     purpose: 'Full-text extraction from web sources' },
  { component: 'Data handling',        tool: 'pandas',                          purpose: 'CSV processing throughout pipeline' },
  { component: 'Visualisation',        tool: 'matplotlib, Plotly',              purpose: 'Static and interactive figures' },
  { component: 'IRR statistics',       tool: 'Custom Python (Cohen\'s κ)',      purpose: 'Inter-rater reliability analysis' },
  { component: 'Reference management', tool: 'EPPI Reviewer',                   purpose: 'Human screening and RIS exports' },
  { component: 'Word documents',       tool: 'python-docx',                     purpose: 'Automated methodology appendix generation' },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function SectionHeading({ label, title, subtitle, dark = false }: { label: string; title: string; subtitle?: string; dark?: boolean }) {
  return (
    <div className="mb-10">
      <p className={`text-xs font-tagline uppercase tracking-widest mb-2 ${dark ? 'text-white/80' : 'text-green'}`}>{label}</p>
      <h2 className={`text-2xl md:text-3xl font-logo font-bold mb-2 ${dark ? 'text-white' : 'text-charcoal'}`}>{title}</h2>
      {subtitle && <p className={`font-tagline max-w-2xl ${dark ? 'text-white/60' : 'text-gray-500'}`}>{subtitle}</p>}
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="text-left px-3 py-2 text-xs font-tagline font-semibold text-gray-500 uppercase tracking-wide border-b border-gray-200">{children}</th>;
}
function Td({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-3 py-2.5 text-xs font-tagline text-charcoal ${className}`}>{children}</td>;
}

// ─── Page ────────────────────────────────────────────────────────────────────

const navItems = [
  { id: 'step1', label: '1 · Search' },
  { id: 'step2', label: '2 · Dedup' },
  { id: 'step3', label: '3 · Calibration' },
  { id: 'step4', label: '4 · Screening' },
  { id: 'step5', label: '5 · Full-text' },
  { id: 'step6', label: '6 · Coding' },
  { id: 'step6c', label: '6c · Saturation' },
  { id: 'step7', label: '7 · Map' },
];

export default function MethodologyPage() {
  const flowRef = useRef<HTMLElement>(null);
  const [navVisible, setNavVisible] = useState(false);

  useEffect(() => {
    const el = flowRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => setNavVisible(!entry.isIntersecting),
      { threshold: 0 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <main className="page-wrapper min-h-screen flex flex-col">
      <Navbar />

      {/* Hero */}
      <section className="relative min-h-[52vh] flex items-end px-6 pt-32 pb-16 text-sand overflow-hidden">
        <video autoPlay muted loop playsInline className="absolute inset-0 w-full h-full object-cover z-[-1]">
          <source src="https://bristlepine.s3.us-east-2.amazonaws.com/2994205-uhd_3840_2160_30fps_clip.mp4" type="video/mp4" />
        </video>
        <div className="absolute inset-0 bg-black/70 z-0" />
        <div className="relative z-10 max-w-4xl mx-auto w-full">
          <p className="text-sm font-tagline uppercase tracking-widest text-white/50 mb-3">Computational Pipeline · Systematic Map</p>
          <h1 className="text-4xl md:text-5xl font-logo font-bold mb-4 leading-tight">Methodology</h1>
          <p className="text-lg font-tagline text-white/70 max-w-3xl leading-relaxed">
            40,634 records across 29 sources. Cross-database deduplication. Six calibration rounds. LLM full-text screening at scale. 8,753 included at title/abstract screening; 2,750 included at full-text screening.
          </p>
        </div>
      </section>

      {/* Key stats bar */}
      <div className="bg-green text-white px-6 py-5">
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          {pipelineStats.map(s => (
            <div key={s.label}>
              <p className="text-2xl font-logo font-bold">{s.value}</p>
              <p className="text-xs font-tagline text-white/70 mt-0.5 uppercase tracking-wide">{s.label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Mini-nav — fixed below main navbar, shown once flow diagram scrolls out of view */}
      <nav
        className="fixed left-0 right-0 z-40 transition-all duration-200"
        style={{
          top: '68px',
          background: 'var(--green)',
          transform: navVisible ? 'translateY(0)' : 'translateY(-110%)',
          opacity: navVisible ? 1 : 0,
          pointerEvents: navVisible ? 'auto' : 'none',
        }}
      >
        <div className="max-w-5xl mx-auto flex items-center justify-center gap-0.5 px-6 py-2 overflow-x-auto">
          {navItems.map(item => (
            <a
              key={item.id}
              href={`#${item.id}`}
              className="px-3 py-1.5 rounded-full text-xs font-tagline font-semibold whitespace-nowrap transition"
              style={{ color: 'rgba(255,255,255,0.75)' }}
              onMouseEnter={e => { (e.target as HTMLElement).style.color = '#fff'; (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.15)'; }}
              onMouseLeave={e => { (e.target as HTMLElement).style.color = 'rgba(255,255,255,0.75)'; (e.target as HTMLElement).style.background = 'transparent'; }}
            >
              {item.label}
            </a>
          ))}
        </div>
      </nav>

      {/* Flow diagram */}
      <section ref={flowRef} className="bg-sand px-6 py-14 border-b border-gray-200">
        <div className="max-w-4xl mx-auto">
          <p className="text-xs font-tagline uppercase tracking-widest text-green mb-10">Pipeline overview</p>

          {/* Row 1: steps 1–4 */}
          <div className="flex items-stretch">
            {[
              { id: 'step1', n: '1',  label: 'Database search',   sub: '40,634 records · 29 sources', color: '#2563eb' },
              { id: 'step2', n: '2',  label: 'Deduplication',      sub: '26,173 unique records',       color: '#4f46e5' },
              { id: 'step3', n: '3',  label: 'Calibration',        sub: '6 rounds · κ = 0.720',        color: '#7c3aed' },
              { id: 'step4', n: '4',  label: 'Abstract screening', sub: '8,753 included',              color: '#9333ea' },
            ].map((s, i, arr) => (
              <div key={s.id} className="flex items-center flex-1 min-w-0">
                <a href={`#${s.id}`} className="flex-1 group rounded-xl border border-gray-200 bg-white shadow-sm px-3 py-3.5 flex flex-col items-center text-center hover:shadow-md transition min-w-0">
                  <div className="w-7 h-7 rounded-full flex items-center justify-center text-white font-logo font-bold text-[11px] mb-2 group-hover:scale-110 transition" style={{ background: s.color }}>{s.n}</div>
                  <p className="font-logo font-bold text-charcoal text-[11px] leading-tight mb-0.5">{s.label}</p>
                  <p className="font-tagline text-gray-400 text-[10px] leading-snug">{s.sub}</p>
                </a>
                {i < arr.length - 1 && <div className="shrink-0 px-1 text-gray-300 font-bold text-sm">→</div>}
              </div>
            ))}
          </div>

          {/* Arrow straight down to step 5 */}
          <div className="flex justify-center"><div className="w-px h-8 bg-gray-300" /></div>

          {/* Row 2: step 5 centered */}
          <div className="flex justify-center">
            <a href="#step5" className="w-44 group rounded-xl border border-gray-200 bg-white shadow-sm px-3 py-3.5 flex flex-col items-center text-center hover:shadow-md transition">
              <div className="w-7 h-7 rounded-full flex items-center justify-center text-white font-logo font-bold text-[11px] mb-2 group-hover:scale-110 transition" style={{ background: '#0d9488' }}>5</div>
              <p className="font-logo font-bold text-charcoal text-[11px] leading-tight mb-0.5">Full-text retrieval</p>
              <p className="font-tagline text-gray-400 text-[10px] leading-snug">3,505 full texts</p>
            </a>
          </div>

          {/* Fork SVG: down from centre, split out to 6a (left) and 6b (right) */}
          <svg className="w-full" height="44" style={{ display: 'block' }}>
            <line x1="50%" y1="0"  x2="50%" y2="22" stroke="#d1d5db" strokeWidth="1.5" />
            <line x1="25%" y1="22" x2="75%" y2="22" stroke="#d1d5db" strokeWidth="1.5" />
            <line x1="25%" y1="22" x2="25%" y2="44" stroke="#d1d5db" strokeWidth="1.5" />
            <line x1="75%" y1="22" x2="75%" y2="44" stroke="#d1d5db" strokeWidth="1.5" />
          </svg>

          {/* Row 3: 6a and 6b, spread to match SVG endpoints */}
          <div className="flex gap-4">
            <div className="flex-1 flex justify-center">
              <a href="#step6a" className="w-44 group rounded-xl border border-gray-200 bg-white shadow-sm px-3 py-3.5 flex flex-col items-center text-center hover:shadow-md transition">
                <div className="w-7 h-7 rounded-full flex items-center justify-center text-white font-logo font-bold text-[11px] mb-2 group-hover:scale-110 transition" style={{ background: '#ea580c' }}>6a</div>
                <p className="font-logo font-bold text-charcoal text-[11px] leading-tight mb-0.5">Human coding</p>
                <p className="font-tagline text-gray-400 text-[10px] leading-snug">Batches of 20 · ongoing</p>
              </a>
            </div>
            <div className="flex-1 flex justify-center">
              <a href="#step6b" className="w-44 group rounded-xl border border-gray-200 bg-white shadow-sm px-3 py-3.5 flex flex-col items-center text-center hover:shadow-md transition">
                <div className="w-7 h-7 rounded-full flex items-center justify-center text-white font-logo font-bold text-[11px] mb-2 group-hover:scale-110 transition" style={{ background: '#d97706' }}>6b</div>
                <p className="font-logo font-bold text-charcoal text-[11px] leading-tight mb-0.5">LLM coding</p>
                <p className="font-tagline text-gray-400 text-[10px] leading-snug">2,750 included · map toggle</p>
              </a>
            </div>
          </div>

          {/* Merge SVG: both 6a and 6b converge back to centre, then down to 7 */}
          <svg className="w-full" height="44" style={{ display: 'block' }}>
            <line x1="25%" y1="0"  x2="25%" y2="22" stroke="#d1d5db" strokeWidth="1.5" />
            <line x1="75%" y1="0"  x2="75%" y2="22" stroke="#d1d5db" strokeWidth="1.5" />
            <line x1="25%" y1="22" x2="75%" y2="22" stroke="#d1d5db" strokeWidth="1.5" />
            <line x1="50%" y1="22" x2="50%" y2="44" stroke="#d1d5db" strokeWidth="1.5" />
          </svg>

          {/* Row 4: step 7 centered */}
          <div className="flex justify-center">
            <a href="#step7" className="w-44 group rounded-xl bg-white shadow-sm px-3 py-3.5 flex flex-col items-center text-center hover:shadow-md transition" style={{ border: '2px solid var(--green)' }}>
              <div className="w-7 h-7 rounded-full flex items-center justify-center text-white font-logo font-bold text-[11px] mb-2 group-hover:scale-110 transition" style={{ background: 'var(--green)' }}>7</div>
              <p className="font-logo font-bold text-charcoal text-[11px] leading-tight mb-0.5">Systematic map</p>
              <p className="font-tagline text-gray-400 text-[10px] leading-snug">Figures & outputs</p>
            </a>
          </div>
        </div>
      </section>

      {/* 1. Database search */}
      <section id="step1" className="bg-white px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <SectionHeading label="Step 1" title="Database search" />
          <div className="overflow-x-auto rounded-xl border border-gray-200 mb-4">
            <table className="w-full border-collapse text-xs font-tagline">
              <thead className="bg-gray-50">
                <tr><Th>Category</Th><Th>Sources</Th><Th>Records</Th><Th>Abstracts</Th></tr>
              </thead>
              <tbody>
                {[
                  { cat: 'Bibliographic databases', sources: 'Scopus, Web of Science, CAB Abstracts, Academic Search Premier, EconLit, ProQuest, AGRIS', records: '39,949', abs: '39,949' },
                  { cat: 'Web search engines',      sources: 'Google Scholar, DuckDuckGo', records: '196', abs: '196' },
                  { cat: 'UN agencies',             sources: 'FAO, IFAD, UNDP, UNEP, UNFCCC', records: '57', abs: '57' },
                  { cat: 'Development agencies',    sources: 'World Bank, GEF, GCF, IDB, ADB, AfDB, FCDO', records: '338', abs: '338' },
                  { cat: 'Research centres',        sources: 'CGSpace (CGIAR), IPAM, Adaptation Research Alliance, GCA, WASP', records: '86', abs: '86' },
                  { cat: 'M&E networks',            sources: '3ie, Campbell Collaboration, J-PAL', records: '8', abs: '8' },
                ].map((row, i) => (
                  <tr key={row.cat} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}>
                    <Td className="font-semibold">{row.cat}</Td>
                    <Td className="text-gray-500">{row.sources}</Td>
                    <Td>{row.records}</Td>
                    <Td className="text-green font-semibold">{row.abs}</Td>
                  </tr>
                ))}
                <tr className="border-t-2 border-gray-300 bg-charcoal text-white">
                  <td className="px-3 py-2.5 font-logo font-bold text-xs" colSpan={2}>Total — 29 sources</td>
                  <td className="px-3 py-2.5 font-bold text-xs text-white">40,634</td>
                  <td className="px-3 py-2.5 font-bold text-xs text-sand">40,634 (100%)</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs font-tagline text-gray-400">Unrecoverable records removed before deduplication: dead URLs, no abstract section, format issues.</p>
        </div>
      </section>

      {/* 2. Deduplication */}
      <section id="step2" className="bg-sand px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <SectionHeading label="Step 2" title="Deduplication" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {[
              { value: '40,634', label: 'Total records across all 29 sources', color: 'bg-blue-600' },
              { value: '14,461', label: 'Duplicates removed',                  color: 'bg-red-500' },
              { value: '26,173', label: 'Unique records entering screening',   color: 'bg-teal-500' },
              { value: '100%',   label: 'Abstract coverage',                   color: 'bg-green-600' },
            ].map(s => (
              <div key={s.label} className="rounded-xl bg-white border border-gray-100 p-4 flex items-center gap-3">
                <div className={`w-2 h-10 rounded-full shrink-0 ${s.color}`} />
                <div>
                  <p className="text-xl font-logo font-bold text-charcoal">{s.value}</p>
                  <p className="text-xs font-tagline text-gray-500 leading-tight">{s.label}</p>
                </div>
              </div>
            ))}
          </div>
          <p className="font-tagline text-sm text-gray-600 mb-5">All 40,634 records from all 29 sources were pooled and deduplicated together using three passes in priority order. A record is a duplicate if it matches on any pass.</p>
          <div className="overflow-x-auto rounded-xl border border-gray-200">
            <table className="w-full border-collapse text-xs font-tagline">
              <thead className="bg-gray-50">
                <tr><Th>Pass</Th><Th>Method</Th><Th>Detail</Th></tr>
              </thead>
              <tbody>
                {[
                  { pass: '1', method: 'DOI match',              detail: 'DOIs normalised: lowercased, URL prefix stripped. Exact string match.' },
                  { pass: '2', method: 'Exact title + year',      detail: 'Title lowercased, punctuation stripped, whitespace collapsed. Both title and year must match.' },
                  { pass: '3', method: 'Fuzzy title (same year)', detail: 'Jaccard token overlap ≥ 0.85 within the same year. Skipped if title has fewer than 4 tokens to avoid false positives.' },
                ].map((row, i) => (
                  <tr key={row.pass} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}>
                    <Td className="font-semibold text-center">{row.pass}</Td>
                    <Td className="font-semibold">{row.method}</Td>
                    <Td className="text-gray-600">{row.detail}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs font-tagline text-gray-400">Priority order ensures the most precise match is used first. Pass 3 catches minor title variations (subtitles, punctuation, truncation) without risking false positives across different years.</p>
        </div>
      </section>

      {/* 3. Calibration */}
      <section id="step3" className="bg-white px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <SectionHeading label="Step 3" title="Calibration & eligibility criteria development" subtitle="Two independent human reviewers (Caroline Staub, Jennifer Cisse) screened each batch in EPPI Reviewer, reconciled disagreements into a gold standard, then the LLM was assessed against it. Criteria were revised between rounds. Full-corpus screening withheld until sensitivity ≥ 0.95 and κ ≥ 0.60." />
          <div className="overflow-x-auto rounded-xl border border-gray-200">
            <table className="w-full border-collapse">
              <thead className="bg-gray-50">
                <tr>
                  <Th>Round</Th><Th>n</Th><Th>Sensitivity</Th><Th>Specificity</Th>
                  <Th>Precision</Th><Th>F1</Th><Th>LLM κ</Th><Th>Human κ</Th><Th>Passed</Th>
                </tr>
              </thead>
              <tbody>
                {benchmarks.map((b) => (
                  <tr key={b.label} className="bg-blue-50/40 border-b border-blue-100">
                    <Td className="text-blue-600 italic">{b.label}</Td>
                    <Td className="text-blue-600 italic">—</Td>
                    <Td className="text-blue-600 italic font-semibold">{b.sens}</Td>
                    <Td className="text-blue-600 italic">{b.spec}</Td>
                    <Td className="text-blue-600 italic">{b.prec}</Td>
                    <Td className="text-blue-600 italic">{b.f1}</Td>
                    <Td className="text-blue-600 italic">{b.kappa}</Td>
                    <Td className="text-blue-600 italic">{b.humanK}</Td>
                    <Td>{''}</Td>
                  </tr>
                ))}
                <tr><td colSpan={9} className="h-px bg-gray-300" /></tr>
                {calibrationRounds.map((r) => (
                  <tr key={r.round} className={r.pass ? 'bg-green/5' : 'bg-white'}>
                    <Td className={r.pass ? 'font-semibold text-green' : ''}>{r.round}</Td>
                    <Td>{r.n}</Td>
                    <Td className={parseFloat(r.sens) >= 0.95 ? 'font-bold text-green' : ''}>{r.sens}</Td>
                    <Td>{r.spec}</Td><Td>{r.prec}</Td><Td>{r.f1}</Td>
                    <Td className={parseFloat(r.kappa) >= 0.60 ? 'font-bold text-green' : ''}>{r.kappa}</Td>
                    <Td>{r.humanK}</Td>
                    <Td>{r.pass
                      ? <span className="inline-block px-2 py-0.5 rounded-full bg-green text-white text-xs font-tagline font-semibold">✓ Yes</span>
                      : <span className="inline-block px-2 py-0.5 rounded-full bg-gray-100 text-gray-400 text-xs font-tagline">Refining</span>}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 space-y-1 text-xs font-tagline text-gray-400">
            <p>R3a: independent 107-paper sample with new gold standard; confirms stability of R2b criteria (1 miss in 33 true positives).</p>
            <p>R2b 95% CI (Wilson): 0.828–0.994. R3a: 0.847–0.995. Pooled (60/62 true positives): 0.890–0.991.</p>
          </div>
        </div>
      </section>

      {/* 4. Abstract screening */}
      <section id="step4" className="bg-sand px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <SectionHeading label="Step 4" title="Title/abstract screening" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {[
              { value: '26,173', label: 'Records screened', color: 'bg-blue-600' },
              { value: '8,753',  label: 'Included (Scopus 6,218 + multi 2,535)', color: 'bg-green-600' },
              { value: '17,420', label: 'Excluded',         color: 'bg-red-500' },
              { value: '0',      label: 'Missing abstract', color: 'bg-gray-400' },
            ].map(s => (
              <div key={s.label} className="rounded-xl bg-white border border-gray-100 p-4 flex items-center gap-3">
                <div className={`w-2 h-10 rounded-full shrink-0 ${s.color}`} />
                <div>
                  <p className="text-xl font-logo font-bold text-charcoal">{s.value}</p>
                  <p className="text-xs font-tagline text-gray-500 leading-tight">{s.label}</p>
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs font-tagline text-gray-500">LLM: qwen2.5:14b via Ollama, temperature 0.0. Per-criterion decisions (yes/no/unclear) with quoted supporting passage. Unverifiable quotes downgraded to unclear. Uncertain or missing defaults to include.</p>
        </div>
      </section>

      {/* 5. Full-text retrieval */}
      <section id="step5" className="bg-white px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <SectionHeading label="Step 5" title="Full-text retrieval" subtitle="Full texts retrieved for all included records via automated API chain. Records from organisational website sources were already held locally as PDFs from the database search stage." />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {[
              { value: '8,748',  label: 'Full texts sought (all databases)', color: 'bg-blue-600' },
              { value: '3,505',  label: 'Retrieved (40.1%)',                 color: 'bg-green-600' },
              { value: '5,243',  label: 'Not retrieved (paywalled / no DOI)', color: 'bg-red-500' },
              { value: '100%',   label: 'Abstract coverage entering FT stage', color: 'bg-purple-600' },
            ].map(s => (
              <div key={s.label} className="rounded-xl bg-white border border-gray-100 p-4 flex items-center gap-3">
                <div className={`w-2 h-10 rounded-full shrink-0 ${s.color}`} />
                <div>
                  <p className="text-xl font-logo font-bold text-charcoal">{s.value}</p>
                  <p className="text-xs font-tagline text-gray-500 leading-tight">{s.label}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="overflow-x-auto rounded-xl border border-gray-200 mb-4">
            <table className="w-full border-collapse text-xs font-tagline">
              <thead className="bg-gray-50">
                <tr><Th>Source</Th><Th>Records retrieved</Th><Th>Format</Th></tr>
              </thead>
              <tbody>
                {[
                  { src: 'Unpaywall',              n: '2,173', fmt: 'PDF' },
                  { src: 'Elsevier full-text API', n: '1,100', fmt: 'PDF / HTML' },
                  { src: 'Semantic Scholar',       n: '157',   fmt: 'PDF' },
                  { src: 'OpenAlex',               n: '40',    fmt: 'PDF' },
                  { src: 'CORE',                   n: '33',    fmt: 'PDF' },
                  { src: 'Other',                  n: '2',     fmt: 'HTML' },
                ].map((row, i) => (
                  <tr key={row.src} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}>
                    <Td className="font-semibold">{row.src}</Td>
                    <Td>{row.n}</Td>
                    <Td className="text-gray-500">{row.fmt}</Td>
                  </tr>
                ))}
                <tr className="border-t-2 border-gray-300 bg-charcoal text-white">
                  <td className="px-3 py-2.5 font-logo font-bold text-xs">Total retrieved</td>
                  <td className="px-3 py-2.5 font-bold text-xs text-white">3,505</td>
                  <td className="px-3 py-2.5 text-xs text-sand">PDF / HTML</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs font-tagline text-gray-400">Non-retrieved records are retained for inclusion by default — absence of full text is not grounds for exclusion.</p>
        </div>
      </section>

      {/* 6. Full-text screening & coding (6a human + 6b LLM) */}
      <section id="step6" className="relative px-6 py-16 overflow-hidden">
        <video autoPlay muted loop playsInline className="absolute inset-0 w-full h-full object-cover z-[-1]">
          <source src="https://bristlepine.s3.us-east-2.amazonaws.com/2994205-uhd_3840_2160_30fps_clip.mp4" type="video/mp4" />
        </video>
        <div className="absolute inset-0 bg-black/75 z-0" />
        <div className="relative z-10 max-w-5xl mx-auto">
          <SectionHeading dark label="Step 6" title="Full-text screening & data extraction" />

          {/* Randomisation panel */}
          <div className="rounded-xl border border-white/20 p-6 mb-6" style={{ background: 'rgba(255,255,255,0.10)' }}>
            <p className="text-[10px] font-tagline uppercase tracking-widest text-white/70 mb-1">Randomisation</p>
            <h3 className="font-logo font-bold text-white text-lg mb-3">Batch sampling method</h3>
            <p className="text-sm font-tagline text-white/80 leading-relaxed mb-5">
              Papers are sampled using a pure random draw — no stratification. A fixed integer seed controls each batch: seed 42 draws the first 20, seed 43 the next 20, and so on. Every batch explicitly excludes all DOIs already assigned in prior batches, making overlap impossible by construction. Re-running the draw script with the same seed always returns the identical sample, so every batch is fully reproducible and auditable.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { prop: 'Method',       val: 'Pure random (no stratification)' },
                { prop: 'Batch size',   val: '20 papers' },
                { prop: 'Seeds',        val: '42, 43, 44, … (one per batch)' },
                { prop: 'No-overlap',   val: 'Prior-batch DOIs excluded before each draw' },
              ].map(item => (
                <div key={item.prop} className="rounded-lg p-3" style={{ background: 'rgba(255,255,255,0.08)' }}>
                  <p className="text-[10px] font-tagline uppercase tracking-widest text-white/50 mb-1">{item.prop}</p>
                  <p className="text-xs font-tagline text-white font-semibold leading-snug">{item.val}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* 6a — Human coders */}
            <div id="step6a" className="rounded-xl border border-white/20 p-6" style={{ background: 'rgba(255,255,255,0.12)', borderLeft: '4px solid var(--green)' }}>
              <p className="text-[10px] font-tagline uppercase tracking-widest text-white/70 mb-1">6a</p>
              <h3 className="font-logo font-bold text-white mb-4">Human coders</h3>
              <table className="w-full text-xs font-tagline mb-5">
                <tbody>
                  {[
                    { label: 'Pool',             value: '8,753 included records (T&A screening)' },
                    { label: 'Target',           value: 'Batches drawn until convergence (few hundred papers), not full pool' },
                    { label: 'Fields extracted', value: '20 (country, scale, producer type, adaptation domain, intervention, outcome, effectiveness metric, equity, study design, evidence quality, …)' },
                    { label: 'Status',           value: 'Ongoing' },
                  ].map(row => (
                    <tr key={row.label} className="border-b border-white/10">
                      <td className="py-2 pr-4 text-white/90 w-36 align-top">{row.label}</td>
                      <td className="py-2 text-white font-semibold">{row.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* 6b — LLM parallel track */}
            <div id="step6b" className="rounded-xl border border-white/20 p-6" style={{ background: 'rgba(255,255,255,0.12)', borderLeft: '4px solid #c2703a' }}>
              <p className="text-[10px] font-tagline uppercase tracking-widest text-white/70 mb-1">6b</p>
              <h3 className="font-logo font-bold text-white mb-4">LLM — automated parallel track</h3>
              <table className="w-full text-xs font-tagline">
                <tbody>
                  {[
                    { label: 'Full texts retrieved',       value: '3,505 of 8,748 sought (40%)' },
                    { label: 'Included after FT screening', value: '2,750' },
                    { label: 'Excluded after FT screening', value: '673' },
                    { label: 'Fields extracted',           value: 'Same 20-field schema' },
                    { label: 'Display',                    value: 'Toggleable overlay on systematic map' },
                  ].map(row => (
                    <tr key={row.label} className="border-b border-white/10">
                      <td className="py-2 pr-4 text-white/90 w-36 align-top">{row.label}</td>
                      <td className="py-2 text-white font-semibold">{row.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

          </div>
        </div>
      </section>

      {/* 6c. Information saturation */}
      <section id="step6c" className="bg-white px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <SectionHeading
            label="Step 6c"
            title="Information saturation"
            subtitle="Human coding continues until the distribution of key evidence characteristics stops changing — not until all papers are coded."
          />
          <p className="font-tagline text-sm text-gray-600 leading-relaxed max-w-3xl mb-6">
            At each batch, we track how many new canonical codebook categories appear across three key dimensions: process/outcome domains (13 canonical values), methodological approaches (5), and producer types (5). Values entered outside the codebook are treated as a single &quot;other&quot; category. As more papers are coded, the rate of discovery drops — eventually reaching zero, indicating that additional coding is unlikely to reveal new evidence structures. In this study, all three dimensions reached saturation by batch FT-R2c (49 papers coded), with zero new canonical categories added across the final two batches (R2d and R3, covering an additional 37 papers). This confirms that the extracted evidence space is well-characterised by the human sample without needing to code the full corpus.
          </p>
          <div className="rounded-xl border border-gray-200 overflow-hidden bg-white shadow-sm">
            <PlotlyChart
              src="/map/data/saturation.json"
              height={520}
            />
          </div>
          <p className="mt-3 text-xs font-tagline text-gray-400">
            Top panel: cumulative unique categories as a percentage of final total, by papers coded. Dashed line marks the 95% saturation threshold. Bottom panel: new categories discovered per batch. All three tracked dimensions plateau by 49 papers.
          </p>
        </div>
      </section>

      {/* 7. Systematic map */}
      <section id="step7" className="bg-sand px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <SectionHeading label="Step 7" title="Systematic map & outputs" />
          <p className="font-tagline text-sm text-gray-600 mb-6">All figures generated programmatically from the coded dataset. Human and LLM coding tracks shown as a toggle on the map page.</p>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {['ROSES flow diagram', 'Evidence gap map', 'Temporal trends', 'Geographic distribution', 'Domain heatmap', 'Equity dimensions'].map(fig => (
              <div key={fig} className="rounded-xl bg-white border border-gray-200 px-4 py-3 flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-green shrink-0" />
                <span className="font-tagline text-xs text-charcoal">{fig}</span>
              </div>
            ))}
          </div>
          <p className="mt-6 text-xs font-tagline text-gray-400">
            <a href="/systematic-map" className="underline hover:text-green">View the systematic map →</a>
          </p>
        </div>
      </section>

      {/* Software */}
      <section className="bg-sand px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <SectionHeading label="Software & dependencies" title="Open source stack" />
          <div className="overflow-x-auto rounded-xl border border-gray-200">
            <table className="w-full border-collapse">
              <thead className="bg-white">
                <tr>
                  <Th>Component</Th>
                  <Th>Library / Service</Th>
                  <Th>Purpose</Th>
                </tr>
              </thead>
              <tbody>
                {softwareStack.map((row, i) => (
                  <tr key={row.component} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}>
                    <Td className="font-semibold">{row.component}</Td>
                    <Td><code className="text-xs bg-gray-100 px-1.5 py-0.5 rounded">{row.tool}</code></Td>
                    <Td>{row.purpose}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Repository link */}
          <div className="mt-8 flex items-center gap-4">
            <a
              href="https://github.com/bristlepine/ilri-climate-adaptation-effectiveness"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm font-tagline font-semibold px-4 py-2 rounded-full bg-charcoal text-sand hover:bg-black transition"
            >
              <ExternalLink className="w-3.5 h-3.5" /> View on GitHub
            </a>
            <p className="text-xs font-tagline text-gray-400">
              All pipeline code, criteria YAML, and calibration data are publicly available.
            </p>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
