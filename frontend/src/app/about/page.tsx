'use client';

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { motion } from "framer-motion";

export default function AboutPage() {
  const README_HTML = `<a href="https://doi.org/10.5281/zenodo.17809739" class="text-green underline">&lt;img alt=&quot;DOI&quot; src=&quot;https://zenodo.org/badge/DOI/10.5281/zenodo.17809739.svg&quot; class=&quot;my-4 rounded&quot; /&gt;</a>

<h1 class="text-3xl font-bold mt-12 mb-6">ILRI – Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector.</h1>
<h3 class="text-xl font-semibold mt-6 mb-2">Evidence synthesis and systematic reviews on climate change and agri-food systems</h3>

<p class="my-3 leading-relaxed">This repository hosts the methods, workflows, protocols, documentation, and outputs for the ILRI evidence-synthesis project assessing the effectiveness of climate adaptation interventions for smallholder producers. The project includes a scoping review, systematic evidence map, and systematic review/meta-analysis, following CEE, Campbell Collaboration, and Cochrane standards.</p>

<p class="my-3 leading-relaxed">---</p>

<h2 class="text-2xl font-bold mt-10 mb-4">Contributors</h2>

<h3 class="text-xl font-semibold mt-6 mb-2">Principal Investigators &amp; Authors  </h3>
<ul class="my-4 list-disc ml-6">
<li>Jennifer Denno Cissé — Bristlepine Resilience Consultants — ORCID: 0000-0001-5637-1941 — jenn@bristlep.com  </li>
<li>Caroline G. Staub — Bristlepine Resilience Consultants — caroline@bristlep.com  </li>
<li>Zarrar Khan — Bristlepine Resilience Consultants — ORCID: 0000-0002-8147-8553 — zarrar@bristlep.com  </li>

</ul>

<h3 class="text-xl font-semibold mt-6 mb-2">Systematic Review Methodology Support  </h3>
<ul class="my-4 list-disc ml-6">
<li>Neal Haddaway — Evidence Synthesis Specialist</li>

</ul>

<h3 class="text-xl font-semibold mt-6 mb-2">Project Coordination  </h3>
<ul class="my-4 list-disc ml-6">
<li>Aditi Mukherji — Principal Scientist, Climate Action — ILRI  </li>

</ul>

<p class="my-3 leading-relaxed">---</p>

<h2 class="text-2xl font-bold mt-10 mb-4">How to Cite This Repository</h2>

<p class="my-3 leading-relaxed">Cissé, J. D., Staub, C. G., &amp; Khan, Z. (2025). ILRI – Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector. Version 1.0. Zenodo. https://doi.org/10.5281/zenodo.17809739</p>

<p class="my-3 leading-relaxed">Each deliverable will receive its own versioned DOI, listed below.</p>

<p class="my-3 leading-relaxed">---</p>

<h2 class="text-2xl font-bold mt-10 mb-4">Repository &amp; Output Locations</h2>

<p class="my-3 leading-relaxed">| Platform | Purpose | URL / Identifier |</p>
<p class="my-3 leading-relaxed">|----------|---------|------------------|</p>
<p class="my-3 leading-relaxed">| &lt;strong&gt;GitHub&lt;/strong&gt; | Code, methods, workflows, documentation | https://github.com/bristlepine/ilri-climate-adaptation-effectiveness |</p>
<p class="my-3 leading-relaxed">| &lt;strong&gt;Zenodo&lt;/strong&gt; | DOI-minted snapshots of releases | https://doi.org/10.5281/zenodo.17809739 |</p>
<p class="my-3 leading-relaxed">| &lt;strong&gt;CGSpace&lt;/strong&gt; | Permanent ILRI archive for final outputs | Handle: To be added |</p>
<p class="my-3 leading-relaxed">| &lt;strong&gt;Journal Publications&lt;/strong&gt; | Scoping review, systematic map, systematic review | To be added |</p>

<p class="my-3 leading-relaxed">---</p>

<h2 class="text-2xl font-bold mt-10 mb-4">Deliverables Summary (Aligned to Contract)</h2>

<p class="my-3 leading-relaxed">| No. | Deliverable | Type | Due Date | Status | DOI |</p>
<p class="my-3 leading-relaxed">|-----|-------------|------|----------|--------|------|</p>
<p class="my-3 leading-relaxed">| 1 | Inception Report | Final RQs, search plan, Gantt chart | Interim | Submitted | &lt;a href=&quot;https://github.com/bristlepine/ilri-climate-adaptation-effectiveness/blob/main/deliverables/01_inception_report/Deliverable%201_Inception%20Report_IL01_v1.pdf&quot; class=&quot;text-green underline&quot;&gt;PDF&lt;/a&gt; • &lt;a href=&quot;https://doi.org/10.5281/zenodo.17861055&quot; class=&quot;text-green underline&quot;&gt;DOI&lt;/a&gt; |</p>
<p class="my-3 leading-relaxed">| 2 | Draft Scoping Review &amp; Systematic Map Protocol | Interim Report | Jan 2, 2026 | In Progress | TBD |</p>
<p class="my-3 leading-relaxed">| 3 | Final Scoping Review / Systematic Map Protocol (CGSpace) | Final Report | Jan 30, 2026 | Not Started | TBD |</p>
<p class="my-3 leading-relaxed">| 4 | Draft Scoping Review + Evidence Database | Interim | Feb 27, 2026 | Not Started | TBD |</p>
<p class="my-3 leading-relaxed">| 5 | Final Scoping Review + Systematic Map + Database | Final | Mar 27, 2026 | Not Started | TBD |</p>
<p class="my-3 leading-relaxed">| 6 | Draft Systematic Review / Meta-analysis Protocol | Interim | May 1, 2026 | Not Started | TBD |</p>
<p class="my-3 leading-relaxed">| 7 | Final SR/MA Protocol (CGSpace) | Final | May 29, 2026 | Not Started | TBD |</p>
<p class="my-3 leading-relaxed">| 8 | Draft Systematic Review / Meta-analysis Manuscript | Interim | Jun 26, 2026 | Not Started | TBD |</p>
<p class="my-3 leading-relaxed">| 9 | Final Systematic Review / Meta-analysis Manuscript | Final | Jul 31, 2026 | Not Started | TBD |</p>
<p class="my-3 leading-relaxed">| 10 | Final Stakeholder Presentation | Final | Jul 31, 2026 | Not Started | TBD |</p>

<p class="my-3 leading-relaxed">---</p>

<h2 class="text-2xl font-bold mt-10 mb-4">Repository Structure</h2>


<p class="my-3 leading-relaxed"><pre class="p-4 bg-gray-900 text-gray-100 rounded-md my-4 overflow-auto font-mono text-sm"><code>bash
repo-root/
├── CITATION.cff
├── LICENSE
├── autodocs.py
├── deliverables/
│   ├── 01_inception_report/
│   │   ├── Deliverable 1_Inception Report_IL01_v1.pdf
│   │   ├── README.md
│   │   ├── metadata.json
├── frontend/
│   ├── README.md
│   ├── next-env.d.ts
│   ├── next.config.ts
│   ├── package-lock.json
│   ├── package.json
│   ├── postcss.config.mjs
│   ├── tsconfig.json
</code></pre></p>


<p class="my-3 leading-relaxed">---</p>
`;
  return (
    <main className="page-wrapper">
      <Navbar />

      {/* Hero Section */}
      <section className="relative h-[60vh] w-full text-white overflow-hidden">
        <video autoPlay muted loop playsInline className="absolute inset-0 w-full h-full object-cover z-[-1]">
          <source src="https://bristlepine.s3.us-east-2.amazonaws.com/2994205-uhd_3840_2160_30fps_clip.mp4" type="video/webm" />
        </video>
        <div className="absolute inset-0 bg-black/60 z-0" />
        <div className="relative z-10 flex flex-col justify-center items-center text-center px-6 pt-20 h-full">
          <h1 className="text-5xl font-logo text-sand mb-4">About</h1>
          <p className="text-lg font-tagline text-white/90 max-w-2xl mx-auto">Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector.</p>
        </div>
      </section>

      {/* README Content */}
      <section className="px-6 py-16 bg-white text-charcoal">
        <div className="max-w-4xl mx-auto prose prose-lg">
          <div dangerouslySetInnerHTML={{ __html: README_HTML }} />
        </div>
      </section>

      <Footer />
    </main>
  );
}
