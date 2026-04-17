'use client';

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { ExternalLink } from "lucide-react";

// ─── Data ────────────────────────────────────────────────────────────────────

const pipelineStats = [
  { value: '17,021', label: 'Records screened' },
  { value: '0.966', label: 'Sensitivity (R2b)' },
  { value: 'κ = 0.720', label: 'LLM–human agreement' },
  { value: '6 rounds', label: 'Calibration iterations' },
];

const innovations = [
  {
    title: 'Calibration-first validation',
    body: 'Full-corpus screening did not proceed until the LLM screener met published thresholds across three independent calibration rounds. Two human reviewers screened each calibration batch independently in EPPI Reviewer, reconciled disagreements into a gold standard, then assessed the LLM against it — with criteria revised between rounds.',
  },
  {
    title: 'Conservative inclusion by design',
    body: 'At every screening stage, uncertainty defaults to inclusion rather than exclusion. Records with missing abstracts, ambiguous evidence, or unclear LLM decisions are retained. This encodes the fundamental systematic review principle that missing a relevant study is a more serious error than including an irrelevant one.',
  },
  {
    title: 'Deterministic, locally hosted LLM',
    body: 'All LLM inference uses qwen2.5:14b via Ollama at temperature 0.0 — fully deterministic and reproducible without external API dependencies. Fixed model, fixed input, identical output. Every decision is cached to disk and auditable. The model\'s parameters are never updated; calibration validates prompt and criteria design, not model training.',
  },
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

const pipelineSteps = [
  { step: '01', title: 'Search query design',        desc: 'PCCM-structured query defined in version-controlled YAML. Scopus API queried to validate record counts per element and combination.' },
  { step: '02', title: 'Record retrieval',            desc: 'Full corpus retrieved via Scopus API. 5,000-record paging limit handled by automatic year-slicing. 17,083 records → 17,021 after deduplication (DOI → title+year → title → EID priority).' },
  { step: '03–04', title: 'Benchmark coverage',      desc: 'Pre-compiled list of known key studies enriched with DOIs via Crossref, OpenAlex, Semantic Scholar. Coverage validated; missing records used to refine query.' },
  { step: '08', title: 'Cleaning & deduplication',   desc: 'Deterministic cleaning: HTML unescaping, whitespace normalisation, DOI canonicalisation, year extraction. Missing fields repaired via Crossref. Fully idempotent.' },
  { step: '09–09a', title: 'Abstract enrichment',    desc: 'Missing abstracts retrieved via sequential API chain: Elsevier → Semantic Scholar → OpenAlex → Crossref → Unpaywall → scrape. RIS exports from EPPI Reviewer supplement remaining gaps. 30-day response cache.' },
  { step: '10–11', title: 'Calibration rounds',      desc: 'Six rounds of dual human screening, gold-standard reconciliation, LLM assessment, and criteria revision. Full-corpus screening withheld until all five metrics met published thresholds.' },
  { step: '12', title: 'Title/abstract screening',   desc: 'LLM screener applied to all 17,021 records. Per-criterion decisions (yes/no/unclear) with quoted supporting passages. Unverifiable quotations downgraded to unclear. Any unclear defaults to inclusion. Result: 6,206 included.' },
  { step: '13', title: 'Full-text retrieval',         desc: 'Full texts retrieved for all included records via Unpaywall, Elsevier Full-Text API, Semantic Scholar, OpenAlex, CORE. 4,002 of 6,218 records retrieved (64.4%). Downloads capped at 25 MB.' },
  { step: '14', title: 'Full-text screening',         desc: 'LLM screener applied to retrieved full texts (truncated to 12,000 tokens). Records without full text retained for inclusion by default per Cochrane guidance.' },
  { step: '15', title: 'Data extraction & coding',   desc: '20-field extraction schema: publication year/type, country, geographic scale, producer type, adaptation domain, methodological approach, effectiveness metric, equity dimensions. Coding source tracked per record.' },
  { step: '16', title: 'Systematic map outputs',      desc: 'All publication-ready figures generated programmatically from the coded dataset: ROSES flow diagram, evidence gap map, temporal trends, geographic distribution, domain heatmap.' },
];

const efficiencyData = [
  {
    stage: 'Title/abstract screening',
    n: '17,021',
    manualPerRecord: '4 min\n(2 min × 2 reviewers)',
    manualTotal: '1,135 person-hr',
    computePerRecord: '~0.65 sec',
    computeTotal: '3 hr 5 min',
    saved: '~1,132 person-hr',
  },
  {
    stage: 'Full-text retrieval',
    n: '6,218',
    manualPerRecord: '15 min\n(locate, access, download)',
    manualTotal: '1,552 person-hr',
    computePerRecord: '~6 sec',
    computeTotal: '10 hr 47 min',
    saved: '~1,541 person-hr',
  },
  {
    stage: 'Full-text screening',
    n: '6,206',
    manualPerRecord: '20 min\n(10 min × 2 reviewers)',
    manualTotal: '2,069 person-hr',
    computePerRecord: '~44 sec †',
    computeTotal: '3 hr 50 min †',
    saved: '~2,065 person-hr',
  },
  {
    stage: 'Data extraction & coding',
    n: '6,076',
    manualPerRecord: '25 min\n(1 coder)',
    manualTotal: '2,532 person-hr',
    computePerRecord: '~5 sec ‡',
    computeTotal: '~8 hr est. ‡',
    saved: '~2,524 person-hr',
  },
];

const reproducibilityPrinciples = [
  { title: 'Deterministic outputs',          desc: 'Temperature 0.0 for all LLM calls. Fixed model + fixed input = identical output. No stochastic variation between runs.' },
  { title: 'Comprehensive caching',          desc: 'Every external API response and LLM decision cached to disk. Re-runs process only new or expired records; prior decisions are preserved.' },
  { title: 'Quotation verification',         desc: 'Screening decisions must cite a passage from the abstract. Unverifiable citations are automatically downgraded to \'unclear\' — a lightweight hallucination check built into the pipeline.' },
  { title: 'Conservative defaults',          desc: 'Absence of required evidence defaults to inclusion at all stages. Records with missing abstracts, failed retrievals, or uncertain LLM decisions are never silently excluded.' },
  { title: 'Iterative human calibration',    desc: 'Six calibration rounds with two independent human reviewers. Criteria revised and versioned between rounds. Full-corpus screening withheld until thresholds met.' },
  { title: 'Version-controlled criteria',    desc: 'Eligibility criteria stored in criteria.yml and extraction criteria in criteria_sysmap_v1.yml, versioned in Git alongside all code. Criteria changes are traceable.' },
  { title: 'ROSES compliance',               desc: 'Record counts at every pipeline stage are automatically compiled into a ROSES flow diagram at Step 16, providing a machine-generated audit trail of the entire review process.' },
  { title: 'Coding source provenance',       desc: 'Every coded record carries a coding_source field (full text / abstract only / title-only) so downstream analyses can weight or filter by evidence quality.' },
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

function SectionHeading({ label, title, subtitle }: { label: string; title: string; subtitle?: string }) {
  return (
    <div className="mb-10">
      <p className="text-xs font-tagline uppercase tracking-widest text-green mb-2">{label}</p>
      <h2 className="text-2xl md:text-3xl font-logo font-bold text-charcoal mb-2">{title}</h2>
      {subtitle && <p className="font-tagline text-gray-500 max-w-2xl">{subtitle}</p>}
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

export default function MethodologyPage() {
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
          <p className="text-sm font-tagline uppercase tracking-widest text-clay mb-3">Computational Pipeline · Systematic Map</p>
          <h1 className="text-4xl md:text-5xl font-logo font-bold mb-4 leading-tight">Methodology</h1>
          <p className="text-lg font-tagline text-clay max-w-3xl leading-relaxed">
            A sixteen-step computational pipeline for AI-assisted systematic mapping, combining local large language model inference with structured human calibration and conservative inclusion-by-default logic.
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

      {/* Overview */}
      <section className="bg-white px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <SectionHeading label="Overview" title="Human-guided automation at scale" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-10 font-tagline text-charcoal leading-relaxed text-sm">
            <div>
              <p className="mb-4">
                The pipeline consists of sixteen sequential steps implemented in Python, each producing auditable outputs that feed the next stage. It is fully resumable: all external API responses and LLM decisions are cached to disk, and re-runs process only new or changed records.
              </p>
              <p>
                The design principle is assistive automation — human judgement concentrated at every consequential decision point, with automated processing handling scale. The LLM is a screener, not a decision-maker; human reviewers set the eligibility criteria, validate performance against a reconciled gold standard, and retain final authority over inclusion decisions.
              </p>
            </div>
            <div>
              <p className="mb-4">
                The search was conducted across Scopus (pilot corpus), with Web of Science, CAB Abstracts, AGRIS, and Academic Search Premier integration underway. All LLM steps use qwen2.5:14b via Ollama at temperature 0.0 — fully deterministic and locally hosted, with no external API dependency for inference.
              </p>
              <p>
                Eligibility criteria are stored in version-controlled YAML files alongside the code. Every criteria revision is traceable in the Git history, and the pipeline can be re-run end-to-end against updated criteria at any time.
              </p>
            </div>
          </div>

          {/* Human oversight points */}
          <div className="mt-10 grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { title: 'Search design', desc: 'Query and PCCM eligibility criteria constructed and iteratively refined by the research team before any automated step.' },
              { title: 'Dual-reviewer gold standard', desc: 'Two independent human reviewers (Caroline Staub, Jennifer Cisse) screened each calibration sample and reconciled all disagreements before the LLM was assessed.' },
              { title: 'Spot-checking at every stage', desc: 'Random samples of LLM decisions reviewed at abstract screening, full-text screening, and data extraction. Extracted fields checked against source documents.' },
            ].map(p => (
              <div key={p.title} className="rounded-xl border border-gray-100 bg-gray-50 p-5">
                <h3 className="font-logo font-bold text-charcoal text-sm mb-2">{p.title}</h3>
                <p className="font-tagline text-xs text-gray-600 leading-relaxed">{p.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Key innovations */}
      <section className="bg-sand px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <SectionHeading label="Design principles" title="Three core methodological innovations" />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {innovations.map((inn, i) => (
              <div key={inn.title} className="rounded-xl bg-white border border-gray-200 p-6 flex flex-col">
                <div className="w-8 h-8 rounded-full bg-green text-white flex items-center justify-center font-logo font-bold text-sm mb-4 shrink-0">
                  {i + 1}
                </div>
                <h3 className="font-logo font-bold text-charcoal mb-2">{inn.title}</h3>
                <p className="font-tagline text-xs text-gray-600 leading-relaxed flex-1">{inn.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Calibration — the core story */}
      <section className="bg-white px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <SectionHeading
            label="Calibration & validation"
            title="Six rounds of structured criteria refinement"
            subtitle="Full-corpus screening did not proceed until all five metrics met their published benchmarks. The table below shows the progression from R1 (initial criteria) to R3a (independent stability check on the same criteria as R2b)."
          />

          {/* Metric definitions */}
          <div className="mb-8 overflow-x-auto rounded-xl border border-gray-200">
            <table className="w-full border-collapse">
              <thead className="bg-gray-50">
                <tr>
                  <Th>Metric</Th>
                  <Th>What it measures</Th>
                  <Th>Formula</Th>
                  <Th>Priority</Th>
                </tr>
              </thead>
              <tbody>
                {[
                  { m: 'Sensitivity / Recall', w: 'Of all truly relevant records, what proportion were correctly included?', f: 'TP / (TP + FN)', p: 'Highest — missing a relevant study is the most serious error in systematic reviewing' },
                  { m: 'Specificity', w: 'Of all truly irrelevant records, what proportion were correctly excluded?', f: 'TN / (TN + FP)', p: 'Secondary — false positives are caught at full-text screening' },
                  { m: 'Precision', w: 'Of all included records, what proportion are truly relevant?', f: 'TP / (TP + FP)', p: 'Secondary' },
                  { m: 'F1', w: 'Harmonic mean of precision and recall. Penalises imbalance.', f: '2PR / (P + R)', p: 'Balanced single score' },
                  { m: "Cohen's κ", w: 'Agreement beyond chance between two raters', f: '(p_o − p_e) / (1 − p_e)', p: 'Standard IRR metric — EPPI Reviewer, Cochrane, Campbell' },
                ].map((row, i) => (
                  <tr key={row.m} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}>
                    <Td className="font-semibold">{row.m}</Td>
                    <Td>{row.w}</Td>
                    <Td className="font-mono text-xs">{row.f}</Td>
                    <Td>{row.p}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Calibration results */}
          <div className="overflow-x-auto rounded-xl border border-gray-200">
            <table className="w-full border-collapse">
              <thead className="bg-gray-50">
                <tr>
                  <Th>Round</Th>
                  <Th>n</Th>
                  <Th>Sensitivity</Th>
                  <Th>Specificity</Th>
                  <Th>Precision</Th>
                  <Th>F1</Th>
                  <Th>LLM κ</Th>
                  <Th>Human κ</Th>
                  <Th>Threshold met?</Th>
                </tr>
              </thead>
              <tbody>
                {/* Benchmarks */}
                {benchmarks.map((b, i) => (
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
                {/* Divider */}
                <tr><td colSpan={9} className="h-px bg-gray-300" /></tr>
                {/* Calibration rounds */}
                {calibrationRounds.map((r) => (
                  <tr key={r.round} className={r.pass ? 'bg-green/5' : 'bg-white'}>
                    <Td className={r.pass ? 'font-semibold text-green' : ''}>{r.round}</Td>
                    <Td>{r.n}</Td>
                    <Td className={parseFloat(r.sens) >= 0.95 ? 'font-bold text-green' : ''}>{r.sens}</Td>
                    <Td>{r.spec}</Td>
                    <Td>{r.prec}</Td>
                    <Td>{r.f1}</Td>
                    <Td className={parseFloat(r.kappa) >= 0.60 ? 'font-bold text-green' : ''}>{r.kappa}</Td>
                    <Td>{r.humanK}</Td>
                    <Td>
                      {r.pass
                        ? <span className="inline-block px-2 py-0.5 rounded-full bg-green text-white text-xs font-tagline font-semibold">✓ Yes</span>
                        : <span className="inline-block px-2 py-0.5 rounded-full bg-gray-100 text-gray-400 text-xs font-tagline">Refining</span>
                      }
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 space-y-1.5 text-xs font-tagline text-gray-400">
            <p>† R2a: metrics at time of initial submission (sensitivity 0.897, below ≥0.95 threshold); eligibility criteria subsequently revised before R2b.</p>
            <p>‡ R3a: same criteria as R2b; separate 107-paper sample with independent reconciled gold standard; confirms stability (n=33 true positives, 1 miss).</p>
            <p>§ Scherbakov et al. 2025: F1 computed from reported sensitivity (0.804) and precision (0.632) — not directly reported in the paper.</p>
            <p>R2b sensitivity 95% CI (Wilson): 0.828–0.994. R3a: 0.847–0.995. Pooled across both independent samples (60/62 true positives): 0.890–0.991. Point estimates of both samples exceed ≥0.95; the O'Mara-Eves threshold is defined as a point-estimate target only.</p>
          </div>

          {/* Relationship to supervised ML */}
          <div className="mt-10 rounded-xl border border-gray-200 bg-gray-50 p-6">
            <h3 className="font-logo font-bold text-charcoal mb-2">Relationship to supervised ML screeners</h3>
            <p className="font-tagline text-sm text-gray-600 leading-relaxed">
              Supervised ML screeners (e.g. EPPI Reviewer, Juno) are classifiers trained from near-scratch on labelled examples and typically require 2,000–7,000 training records before reaching adequate performance. qwen2.5:14b is a pre-trained large language model — its parameters are never updated. The ~515 calibration records across four rounds (R1 through R2a) constitute a <em>validation set</em> for prompt and criteria design, not a training corpus. The analogy in conventional systematic review practice is calibration training: verifying that a reviewer correctly understands the eligibility criteria before independent screening begins.
            </p>
          </div>
        </div>
      </section>

      {/* Pipeline stages */}
      <section className="relative px-6 py-16 overflow-hidden">
        <video autoPlay muted loop playsInline className="absolute inset-0 w-full h-full object-cover z-[-1]">
          <source src="https://bristlepine.s3.us-east-2.amazonaws.com/2994205-uhd_3840_2160_30fps_clip.mp4" type="video/mp4" />
        </video>
        <div className="absolute inset-0 bg-black/75 z-0" />
        <div className="relative z-10 max-w-4xl mx-auto">
          <div className="mb-10">
            <p className="text-xs font-tagline uppercase tracking-widest text-clay mb-2">Pipeline</p>
            <h2 className="text-2xl md:text-3xl font-logo font-bold text-sand mb-2">Sixteen sequential steps</h2>
            <p className="font-tagline text-white/75 text-sm max-w-2xl">Each step produces auditable CSV and JSON outputs. The pipeline is fully resumable — re-runs skip cached results and process only new records.</p>
          </div>
          <div className="space-y-3">
            {pipelineSteps.map((step) => (
              <div key={step.step} className="flex gap-4 rounded-xl bg-white/15 border border-white/20 px-5 py-4 hover:bg-white/20 transition">
                <div className="shrink-0 w-10 text-right">
                  <span className="font-logo font-bold text-white text-sm">{step.step}</span>
                </div>
                <div>
                  <h3 className="font-logo font-bold text-white text-sm mb-0.5">{step.title}</h3>
                  <p className="font-tagline text-xs text-white/75 leading-relaxed">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Efficiency */}
      <section className="bg-sand px-6 py-16">
        <div className="max-w-5xl mx-auto">
          <SectionHeading
            label="Computational efficiency"
            title="~7,300 person-hours of manual work condensed to ~24 wall-clock hours"
            subtitle="Manual estimates use published rates (O'Mara-Eves 2015). Compute times are actual wall-clock durations from pipeline run logs. Both sides shown per record and in full."
          />

          <div className="overflow-x-auto rounded-xl border border-gray-200">
            <table className="w-full border-collapse text-xs font-tagline">
              <thead className="bg-white">
                <tr>
                  <th className="text-left px-3 py-2.5 text-xs font-tagline font-semibold text-gray-500 uppercase tracking-wide border-b border-gray-200">Stage</th>
                  <th className="text-center px-3 py-2.5 text-xs font-tagline font-semibold text-gray-500 uppercase tracking-wide border-b border-gray-200">n</th>
                  <th className="text-center px-3 py-2.5 text-xs font-tagline font-semibold text-gray-400 uppercase tracking-wide border-b border-gray-200 bg-orange-50/40" colSpan={2}>Manual (estimated)</th>
                  <th className="text-center px-3 py-2.5 text-xs font-tagline font-semibold text-green/80 uppercase tracking-wide border-b border-gray-200 bg-green/5" colSpan={2}>Pipeline (actual)</th>
                  <th className="text-center px-3 py-2.5 text-xs font-tagline font-semibold text-gray-500 uppercase tracking-wide border-b border-gray-200">Saved</th>
                </tr>
                <tr className="bg-gray-50">
                  <td className="px-3 py-1.5 border-b border-gray-100" />
                  <td className="px-3 py-1.5 border-b border-gray-100" />
                  <td className="px-3 py-1.5 text-center text-gray-400 text-xs border-b border-gray-100 bg-orange-50/40">Per record</td>
                  <td className="px-3 py-1.5 text-center text-gray-400 text-xs border-b border-gray-100 bg-orange-50/40">Total</td>
                  <td className="px-3 py-1.5 text-center text-green/70 text-xs border-b border-gray-100 bg-green/5">Per record</td>
                  <td className="px-3 py-1.5 text-center text-green/70 text-xs border-b border-gray-100 bg-green/5">Total (wall-clock)</td>
                  <td className="px-3 py-1.5 border-b border-gray-100" />
                </tr>
              </thead>
              <tbody>
                {efficiencyData.map((row, i) => (
                  <tr key={row.stage} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50/40'}>
                    <td className="px-3 py-3 font-semibold text-charcoal">{row.stage}</td>
                    <td className="px-3 py-3 text-center text-gray-500">{row.n}</td>
                    <td className="px-3 py-3 text-center text-gray-500 bg-orange-50/30 whitespace-pre-line">{row.manualPerRecord}</td>
                    <td className="px-3 py-3 text-center text-gray-500 bg-orange-50/30 font-semibold">{row.manualTotal}</td>
                    <td className="px-3 py-3 text-center text-green bg-green/5 font-semibold">{row.computePerRecord}</td>
                    <td className="px-3 py-3 text-center text-green bg-green/5 font-bold">{row.computeTotal}</td>
                    <td className="px-3 py-3 text-center font-semibold text-charcoal">{row.saved}</td>
                  </tr>
                ))}
                {/* Total row */}
                <tr className="border-t-2 border-gray-300 bg-charcoal text-white">
                  <td className="px-3 py-3 font-logo font-bold text-sm" colSpan={2}>Total</td>
                  <td className="px-3 py-3 text-center text-white/60 text-xs">—</td>
                  <td className="px-3 py-3 text-center font-bold">~7,288 person-hr<br/><span className="font-tagline font-normal text-white/60">(~304 person-days)</span></td>
                  <td className="px-3 py-3 text-center text-white/60 text-xs">—</td>
                  <td className="px-3 py-3 text-center font-bold text-green">~24 hr<br/><span className="font-tagline font-normal text-white/60">(overnight run)</span></td>
                  <td className="px-3 py-3 text-center font-bold text-green">~7,260 person-hr</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="mt-4 space-y-1.5 text-xs font-tagline text-gray-400">
            <p>Manual rates: title/abstract screening 2 min/record/reviewer × 2 reviewers; full-text retrieval 15 min/record; full-text screening 10 min/record/reviewer × 2 reviewers; data extraction 25 min/record × 1 coder (conservative; published ranges are wider).</p>
            <p>† Full-text screening: 3 hr 50 min elapsed for 6,206 records, of which only 314 had full text retrieved and were LLM-screened (~44 sec/record); 5,892 passed through near-instantly. The planned overnight re-run against all 4,002 retrieved full texts will take approximately 48 hr.</p>
            <p>‡ Data extraction: compute time estimated from calibration run (~100 records); full-corpus run scheduled overnight.</p>
          </div>
        </div>
      </section>

      {/* Reproducibility */}
      <section className="bg-white px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <SectionHeading
            label="Transparency & reproducibility"
            title="Eight reproducibility commitments"
            subtitle="The pipeline was designed from the outset to produce results that can be independently verified, re-run, and audited at every step."
          />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {reproducibilityPrinciples.map((p) => (
              <div key={p.title} className="rounded-xl border border-gray-100 p-5 bg-gray-50/50">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-green mt-1.5 shrink-0" />
                  <div>
                    <h3 className="font-logo font-bold text-charcoal text-sm mb-1">{p.title}</h3>
                    <p className="font-tagline text-xs text-gray-600 leading-relaxed">{p.desc}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
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
