'use client';

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import PlotlyChart from "@/components/PlotlyChart";
import PrismaFlow from "@/components/PrismaFlow";
import { useState, useEffect } from "react";
import { Download, Maximize2, X } from "lucide-react";

type DatasetMode = 'llm' | 'human' | 'compare';

const DATASET_LABELS: Record<DatasetMode, string> = {
  human:   'Human  (n=86)',
  llm:     'LLM  (n=2,368)',
  compare: 'Compare',
};

const DATASET_DESCS: Record<DatasetMode, string> = {
  human:   'Human-coded primary output — 86 records across 5 coding rounds.',
  llm:     'LLM-screened reference corpus — 2,368 auto-extracted records.',
  compare: 'Human (amber) vs LLM (teal) — % of studies for direct comparison.',
};

const DATASET_MODES: DatasetMode[] = ['human', 'llm', 'compare'];

interface Study {
  doi: string;
  title: string;
  year: string;
  pub_type: string;
  country: string;
  geo_scale: string;
  producer_type: string;
  adaptation_focus: string;
  domain_type: string;
  methodology: string;
  equity: string;
}

export default function SystematicMapPage() {
  const [lightbox, setLightbox] = useState<string | null>(null);
  const [expandedFig, setExpandedFig] = useState<{ src: string; pngSrc?: string; csvSrc?: string; title: string } | null>(null);
  const [studies, setStudies] = useState<Study[]>([]);
  const [studiesLoading, setStudiesLoading] = useState(true);
  const [studiesError, setStudiesError] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortCol, setSortCol] = useState<keyof Study>('year');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [geoView, setGeoView] = useState<'map' | 'bar'>('map');
  const [flowView, setFlowView] = useState<'sankey' | 'prisma'>('prisma');
  const [datasetMode, setDatasetMode] = useState<DatasetMode>('llm');

  const figJson = (name: string) => {
    if (datasetMode === 'human')   return `/map/data/human/${name}.json`;
    if (datasetMode === 'compare') return `/map/data/compare/${name}.json`;
    return `/map/data/${name}.json`;
  };
  const figPng = (name: string) =>
    datasetMode === 'llm' ? `/map/${name}.png` : undefined;

  const staticFigures = [
    { name: 'temporal_trends', fixedCats: true, json: figJson('temporal_trends'),  png: figPng('temporal_trends'),  csv: '/map/data/temporal_trends.csv',  title: 'Publications Over Time',     description: 'Number of included studies per publication year.' },
    { name: 'producer_type',   fixedCats: true,  json: figJson('producer_type'),    png: figPng('producer_type_bar'), csv: '/map/data/producer_type.csv',    title: 'Producer Types',             description: 'Breakdown of studies by agricultural producer type.' },
    { name: 'methodology',     fixedCats: true,  json: figJson('methodology'),      png: figPng('methodology_bar'),   csv: '/map/data/methodology.csv',      title: 'Methodological Approaches',  description: 'Primary methodological design across included studies.' },
    { name: 'domain_type',     fixedCats: true,  json: figJson('domain_type'),      png: figPng('domain_type_bar'),   csv: '/map/data/domain_type.csv',      title: 'Adaptation Domain Type',     description: 'Studies assessing processes, outcomes, or both.' },
    { name: 'equity',          fixedCats: true,  json: figJson('equity'),           png: figPng('equity_bar'),        csv: '/map/data/equity.csv',           title: 'Equity & Inclusion',         description: 'Equity and inclusion dimensions addressed in studies.' },
    { name: 'domain_heatmap',  fixedCats: true,  json: figJson('domain_heatmap'),   png: figPng('domain_heatmap'),    csv: '/map/data/domain_heatmap.csv',   title: 'Process & Outcome Domains',  description: datasetMode === 'compare' ? 'Coverage difference (LLM% − Human%) per domain × producer cell. Teal = LLM leads · amber = Human leads.' : 'Adaptation process and outcome domains by producer type.' },
  ];

  useEffect(() => {
    fetch('/map/data/studies.json')
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((data: Study[]) => { setStudies(data); setStudiesLoading(false); })
      .catch(() => { setStudiesError(true); setStudiesLoading(false); });
  }, []);

  const handleSort = (col: keyof Study) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('asc'); }
  };

  const filteredStudies = studies
    .filter(s => {
      if (!searchTerm.trim()) return true;
      const q = searchTerm.toLowerCase();
      return (
        s.title?.toLowerCase().includes(q) ||
        s.country?.toLowerCase().includes(q) ||
        s.producer_type?.toLowerCase().includes(q) ||
        s.adaptation_focus?.toLowerCase().includes(q) ||
        s.domain_type?.toLowerCase().includes(q) ||
        s.methodology?.toLowerCase().includes(q) ||
        s.year?.includes(q)
      );
    })
    .sort((a, b) => {
      const av = (a[sortCol] || '').toLowerCase();
      const bv = (b[sortCol] || '').toLowerCase();
      if (av < bv) return sortDir === 'asc' ? -1 : 1;
      if (av > bv) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });

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
          <strong>In progress — data extraction underway.</strong> 40,653 records identified across 27 sources → 26,182 after deduplication → 8,748 full texts sought → 3,505 assessed → 2,368 included (LLM) · 86 included (Human). Figures update as records are coded. Final version expected May 2026.
        </p>
      </div>

      {/* ROSES Flow Diagram */}
      <section className="bg-white px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-start justify-between mb-6 gap-4 flex-wrap">
            <div>
              <h2 className="text-2xl font-logo font-bold text-green">ROSES Flow Diagram</h2>
              <p className="font-tagline text-sm text-gray-500 mt-1">Record flow across all 27 sources and screening stages, following ROSES reporting standards.</p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {/* View toggle */}
              <div className="flex rounded-full border border-gray-200 overflow-hidden text-xs font-tagline font-semibold">
                <button
                  onClick={() => setFlowView('prisma')}
                  className={`px-3 py-1.5 transition ${flowView === 'prisma' ? 'bg-green text-white' : 'bg-white text-gray-500 hover:text-green'}`}
                >
                  PRISMA
                </button>
                <button
                  onClick={() => setFlowView('sankey')}
                  className={`px-3 py-1.5 transition border-l border-gray-200 ${flowView === 'sankey' ? 'bg-green text-white' : 'bg-white text-gray-500 hover:text-green'}`}
                >
                  Sankey
                </button>
              </div>
              {flowView === 'sankey' && (
                <button
                  onClick={() => setExpandedFig({ src: '/map/data/roses_flow.json', pngSrc: '/map/roses_flow.png', csvSrc: '/map/data/roses_flow.csv', title: 'ROSES Flow Diagram' })}
                  className="flex items-center gap-1.5 text-xs font-tagline font-semibold px-3 py-1.5 rounded-full border border-gray-200 text-gray-500 hover:border-green hover:text-green transition bg-white"
                >
                  <Maximize2 className="w-3.5 h-3.5" /> Fullscreen
                </button>
              )}
            </div>
          </div>
          <div className="rounded-xl border border-gray-200 overflow-hidden bg-white shadow-sm">
            {flowView === 'sankey' ? (
              <PlotlyChart
                src="/map/data/roses_flow.json"
                fallbackImg="/map/roses_flow.png"
                pngSrc="/map/roses_flow.png"
                csvSrc="/map/data/roses_flow.csv"
                height={540}
              />
            ) : (
              <div className="p-6">
                <PrismaFlow />
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Evidence Gap Map */}
      <section className="bg-sand px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center justify-between gap-4 mb-2">
            <h2 className="text-2xl font-logo font-bold text-green shrink-0">Evidence Gap Map</h2>
            <div className="flex-none flex items-center gap-2">
              <div className="flex rounded-full border border-gray-200 overflow-hidden text-xs font-tagline font-semibold">
                {DATASET_MODES.map((mode, i) => (
                  <button key={mode} onClick={() => setDatasetMode(mode)}
                    className={`px-3 py-1.5 transition ${datasetMode === mode ? 'bg-green text-white' : 'bg-white text-gray-500 hover:text-green'} ${i > 0 ? 'border-l border-gray-200' : ''}`}>
                    {DATASET_LABELS[mode]}
                  </button>
                ))}
              </div>
              <button
                onClick={() => setExpandedFig({ src: figJson('evidence_gap_map'), pngSrc: figPng('evidence_gap_map'), csvSrc: '/map/data/evidence_gap_map.csv', title: 'Evidence Gap Map' })}
                className="flex items-center gap-1.5 text-xs font-tagline font-semibold px-3 py-1.5 rounded-full border border-gray-200 text-gray-500 hover:border-green hover:text-green transition bg-white"
              >
                <Maximize2 className="w-3.5 h-3.5" /> Fullscreen
              </button>
            </div>
          </div>
          <p className="font-tagline text-sm text-gray-500 mb-6">
            {datasetMode === 'compare'
              ? 'Left bubble = Human · right = LLM. Blue = process domains · green = outcome. Grey = evidence gap.'
              : 'Bubble size indicates number of studies per domain–producer-type cell. Grey circles indicate evidence gaps.'}
          </p>
          <div className="rounded-xl border border-gray-200 overflow-hidden bg-white shadow-sm">
            <PlotlyChart
              src={figJson('evidence_gap_map')}
              fallbackImg={figPng('evidence_gap_map')}
              pngSrc={figPng('evidence_gap_map')}
              csvSrc="/map/data/evidence_gap_map.csv"
              height={datasetMode === 'compare' ? 740 : 700}
            />
          </div>
        </div>
      </section>

      {/* Geographic Distribution */}
      <section className="bg-white px-6 py-16">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between gap-4 mb-2">
            <h2 className="text-2xl font-logo font-bold text-green shrink-0">Geographic Distribution</h2>
            <div className="flex-none flex items-center gap-2">
              <div className="flex rounded-full border border-gray-200 overflow-hidden text-xs font-tagline font-semibold">
                {DATASET_MODES.map((mode, i) => (
                  <button key={mode} onClick={() => setDatasetMode(mode)}
                    className={`px-3 py-1.5 transition ${datasetMode === mode ? 'bg-green text-white' : 'bg-white text-gray-500 hover:text-green'} ${i > 0 ? 'border-l border-gray-200' : ''}`}>
                    {DATASET_LABELS[mode]}
                  </button>
                ))}
              </div>
              <div className="flex rounded-full border border-gray-200 overflow-hidden text-xs font-tagline font-semibold">
                <button onClick={() => setGeoView('map')}
                  className={`px-3 py-1.5 transition ${geoView === 'map' ? 'bg-green text-white' : 'bg-white text-gray-500 hover:bg-gray-50'}`}>
                  Map
                </button>
                <button onClick={() => setGeoView('bar')}
                  className={`px-3 py-1.5 transition border-l border-gray-200 ${geoView === 'bar' ? 'bg-green text-white' : 'bg-white text-gray-500 hover:bg-gray-50'}`}>
                  Bar
                </button>
              </div>
              <button
                onClick={() => setExpandedFig(geoView === 'map'
                  ? { src: figJson('geographic_map'), pngSrc: figPng('geographic_map'), csvSrc: '/map/data/geographic_map.csv', title: 'Geographic Distribution — Map' }
                  : { src: figJson('geographic_bar'), pngSrc: figPng('geographic_bar'), csvSrc: '/map/data/geographic_map.csv', title: 'Geographic Distribution — Bar' }
                )}
                className="flex items-center gap-1.5 text-xs font-tagline font-semibold px-3 py-1.5 rounded-full border border-gray-200 text-gray-500 hover:border-green hover:text-green transition bg-white"
              >
                <Maximize2 className="w-3.5 h-3.5" /> Fullscreen
              </button>
            </div>
          </div>
          <p className="font-tagline text-sm text-gray-500 mb-4">Countries by study count. Multi-country studies counted in each country.</p>
          <div className="rounded-xl border border-gray-200 overflow-hidden bg-white shadow-sm">
            {geoView === 'map' ? (
              <PlotlyChart
                src={figJson('geographic_map')}
                fallbackImg={figPng('geographic_map')}
                pngSrc={figPng('geographic_map')}
                csvSrc="/map/data/geographic_map.csv"
                height={datasetMode === 'compare' ? 400 : 500}
              />
            ) : (
              <PlotlyChart
                src={figJson('geographic_bar')}
                fallbackImg={figPng('geographic_bar')}
                pngSrc={figPng('geographic_bar')}
                csvSrc="/map/data/geographic_map.csv"
                height={500}
              />
            )}
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
              <p className="font-tagline text-sm text-clay mt-1">All included studies with key metadata. Search across all fields.</p>
            </div>
            <a href="/map/data/studies.csv" download
              className="shrink-0 flex items-center gap-2 text-xs font-tagline font-semibold px-4 py-2 rounded-full bg-white text-charcoal hover:bg-sand transition">
              <Download className="w-3.5 h-3.5" /> Download CSV
            </a>
          </div>

          {studiesLoading && (
            <div className="rounded-xl border border-white/20 bg-white/10 backdrop-blur-sm p-12 text-center">
              <p className="font-tagline text-sm text-clay animate-pulse">Loading studies…</p>
            </div>
          )}

          {studiesError && !studiesLoading && (
            <div className="rounded-xl border border-white/20 bg-white/10 backdrop-blur-sm p-12 text-center">
              <p className="font-logo text-xl text-sand/70 mb-2">Interactive Table</p>
              <p className="font-tagline text-sm text-clay">
                Full searchable, filterable database will appear here once data extraction is complete (D5.6, target May 2026).<br />
                Download the preliminary CSV above to explore included records now.
              </p>
            </div>
          )}

          {!studiesLoading && !studiesError && studies.length > 0 && (
            <div className="rounded-xl border border-white/20 bg-white/95 overflow-hidden">
              {/* Search */}
              <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
                <input
                  type="text"
                  placeholder="Search by title, country, methodology, producer type…"
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                  className="w-full text-sm font-tagline px-3 py-2 rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-green/30 focus:border-green bg-white"
                />
                <p className="text-xs font-tagline text-gray-400 mt-1.5">
                  Showing {filteredStudies.length.toLocaleString()} of {studies.length.toLocaleString()} studies
                </p>
              </div>

              {/* Table */}
              <div className="overflow-auto max-h-[520px]">
                <table className="w-full text-xs font-tagline border-collapse">
                  <thead className="sticky top-0 bg-gray-100 z-10">
                    <tr>
                      {([['year','Year','w-12'],['title','Title','min-w-[220px]'],['country','Country',''],['producer_type','Producer Type',''],['domain_type','Domain Type',''],['methodology','Methodology','']] as [keyof Study, string, string][]).map(([col, label, extra]) => (
                        <th key={col} onClick={() => handleSort(col)}
                          className={`text-left px-3 py-2 text-gray-600 font-semibold border-b border-gray-200 cursor-pointer select-none hover:bg-gray-200 transition ${extra}`}>
                          {label}{sortCol === col ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filteredStudies.map((study, i) => (
                      <tr key={i} className={`border-b border-gray-100 hover:bg-blue-50/40 transition-colors ${i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}`}>
                        <td className="px-3 py-2 text-gray-500 align-top">{study.year || '—'}</td>
                        <td className="px-3 py-2 align-top">
                          {study.doi ? (
                            <a href={`https://doi.org/${study.doi}`} target="_blank" rel="noopener noreferrer"
                              className="text-blue-600 hover:underline line-clamp-2"
                              onClick={e => e.stopPropagation()}>
                              {study.title || '(no title)'}
                            </a>
                          ) : (
                            <span className="text-gray-700 line-clamp-2">{study.title || '(no title)'}</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-gray-600 align-top">{study.country || '—'}</td>
                        <td className="px-3 py-2 text-gray-600 align-top">{study.producer_type || '—'}</td>
                        <td className="px-3 py-2 text-gray-600 align-top">{study.domain_type || '—'}</td>
                        <td className="px-3 py-2 text-gray-600 align-top">{study.methodology || '—'}</td>
                      </tr>
                    ))}
                    {filteredStudies.length === 0 && (
                      <tr>
                        <td colSpan={6} className="px-3 py-8 text-center text-gray-400">No studies match your search.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Supporting Charts */}
      <section className="bg-sand px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center justify-between gap-4 mb-1">
            <h2 className="text-2xl font-logo font-bold text-green shrink-0">Evidence Map</h2>
            <div className="flex-none flex rounded-full border border-gray-200 overflow-hidden text-xs font-tagline font-semibold">
              {DATASET_MODES.map((mode, i) => (
                <button key={mode} onClick={() => setDatasetMode(mode)}
                  className={`px-3 py-1.5 transition ${datasetMode === mode ? 'bg-green text-white' : 'bg-white text-gray-500 hover:text-green'} ${i > 0 ? 'border-l border-gray-200' : ''}`}>
                  {DATASET_LABELS[mode]}
                </button>
              ))}
            </div>
          </div>
          <p className="font-tagline text-xs text-gray-400 mb-0">{DATASET_DESCS[datasetMode]}</p>
          <p className="font-tagline text-sm text-gray-500 mb-1 mt-3">Use the <Maximize2 className="inline w-3.5 h-3.5 mx-0.5 text-gray-400" /> button to expand any chart for full interactive exploration.</p>
          <div className={`grid gap-6 mt-6 ${datasetMode === 'compare' ? 'grid-cols-1' : 'grid-cols-1 md:grid-cols-2'}`}>
            {staticFigures.map((fig) => (
              <div key={fig.name}
                className="group rounded-xl overflow-hidden bg-white shadow-[0_2px_12px_rgba(202,194,181,0.25)] hover:shadow-[0_4px_16px_rgba(202,194,181,0.4)] transition-all duration-300 flex flex-col">
                <div className="relative overflow-hidden bg-white">
                  {datasetMode === 'compare' && !fig.fixedCats ? (
                    /* Free-form / temporal: two charts side by side */
                    <div className="flex divide-x divide-gray-100">
                      <div className="flex-1 min-w-0 relative">
                        <div className="absolute top-2 left-2 z-10 text-xs font-tagline font-semibold text-orange-700 bg-orange-50 border border-orange-200 px-2 py-0.5 rounded-full pointer-events-none">Human (n=86)</div>
                        <PlotlyChart src={`/map/data/human/${fig.name}.json`} height={300} />
                      </div>
                      <div className="flex-1 min-w-0 relative">
                        <div className="absolute top-2 left-2 z-10 text-xs font-tagline font-semibold text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded-full pointer-events-none">LLM (n=2,368)</div>
                        <PlotlyChart src={`/map/data/${fig.name}.json`} height={300} />
                      </div>
                    </div>
                  ) : (
                    /* Single chart: LLM, Human, or fixed-category compare (grouped bars) */
                    <>
                      <PlotlyChart
                        src={fig.json}
                        fallbackImg={datasetMode === 'llm' ? fig.png : undefined}
                        pngSrc={datasetMode === 'llm' ? fig.png : undefined}
                        csvSrc={fig.csv}
                        height={datasetMode === 'compare' ? 360 : 320}
                        className="w-full"
                      />
                      <button
                        onClick={() => setExpandedFig({ src: fig.json, pngSrc: datasetMode === 'llm' ? fig.png : undefined, csvSrc: fig.csv, title: fig.title })}
                        className="absolute top-2 right-2 p-1.5 rounded-full bg-white/80 backdrop-blur-sm border border-gray-200 text-gray-500 hover:text-green hover:border-green transition opacity-0 group-hover:opacity-100"
                        title="Expand chart"
                      >
                        <Maximize2 className="w-3.5 h-3.5" />
                      </button>
                    </>
                  )}
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

      {/* PNG Lightbox (kept for backward compat, currently unused) */}
      {lightbox && (
        <div className="fixed inset-0 bg-black/95 z-50 flex items-center justify-center cursor-zoom-out"
          onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="Enlarged figure" className="w-screen h-screen object-contain p-4" />
          <span className="absolute top-4 right-5 text-white/50 font-tagline text-xs select-none">click anywhere to close</span>
        </div>
      )}

      {/* Interactive Fullscreen Modal */}
      {expandedFig && (
        <div className="fixed inset-0 bg-black/90 z-50 flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-3 bg-white/5 border-b border-white/10 shrink-0">
            <h3 className="font-logo font-bold text-white text-lg">{expandedFig.title}</h3>
            <div className="flex items-center gap-3">
              {expandedFig.pngSrc && (
                <a href={expandedFig.pngSrc} download onClick={e => e.stopPropagation()}
                  className="flex items-center gap-1 text-xs font-tagline font-semibold px-3 py-1.5 rounded-full border border-white/20 text-white/70 hover:border-white hover:text-white transition">
                  <Download className="w-3 h-3" /> PNG
                </a>
              )}
              {expandedFig.csvSrc && (
                <a href={expandedFig.csvSrc} download onClick={e => e.stopPropagation()}
                  className="flex items-center gap-1 text-xs font-tagline font-semibold px-3 py-1.5 rounded-full border border-white/20 text-white/70 hover:border-white hover:text-white transition">
                  <Download className="w-3 h-3" /> CSV
                </a>
              )}
              <button onClick={() => setExpandedFig(null)}
                className="p-2 rounded-full border border-white/20 text-white/70 hover:border-white hover:text-white transition">
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
          {/* Chart */}
          <div className="flex-1 overflow-hidden p-4">
            <PlotlyChart
              src={expandedFig.src}
              pngSrc={expandedFig.pngSrc}
              csvSrc={expandedFig.csvSrc}
              height={typeof window !== 'undefined' ? Math.floor(window.innerHeight * 0.82) : 700}
              className="w-full h-full"
            />
          </div>
        </div>
      )}

      <Footer />
    </main>
  );
}
