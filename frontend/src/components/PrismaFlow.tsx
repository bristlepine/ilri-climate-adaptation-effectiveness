'use client';

import React from 'react';

const GREEN      = '#21472E';
const LIGHT_GRN  = '#eef4f0';
const MID_GRN    = '#c8dece';
const BOX_BG     = '#ffffff';
const EXC_BG     = '#fef2f2';
const EXC_BDR    = '#fca5a5';
const EXC_TEXT   = '#b91c1c';
const GREY_ARR   = '#9ca3af';
const AMBER      = '#b85c00';
const AMBER_BG   = '#fff8f0';
const AMBER_BDR  = '#f6ad55';
const SUPP_BG    = '#f8fafc';
const SUPP_BDR   = '#94a3b8';

// Layout constants
const W   = 800;
const PX  = 92;
const CX  = PX + 8;        // 100 — center column left edge
const CW  = 320;
const CM  = CX + CW / 2;   // 260 — center column midpoint
const EX  = CX + CW + 28;  // 448 — exclusion column left edge
const EW  = W - EX - 6;    // 346 — exclusion column width
const BH  = 52;
const BH1 = 80;

// Box y-positions
const Y1 = 14;
const Y2 = Y1 + BH1 + 22;       // 116
const P1 = Y2 + BH + 18;        // 186 — end of Identification
const Y3 = P1 + 14;             // 200
const P2 = Y3 + BH + 18;        // 270 — end of Screening
const Y4 = P2 + 14;             // 284
const Y5 = Y4 + BH + 22;        // 358
const P3 = Y5 + BH + 18;        // 428 — end of Eligibility

// Two-track layout for INCLUDED section
const TW     = 148;
const TG     = 16;
const LLM_X  = CX + (CW - 2 * TW - TG) / 2;  // centres both tracks in CW
const LLM_C  = LLM_X + TW / 2;
const HUM_X  = LLM_X + TW + TG;
const HUM_C  = HUM_X + TW / 2;

const Y_FORK = P3 + 14;          // 442 — fork point
const Y_R1   = Y_FORK + 32;      // 474 — Row 1: track overview boxes
const R1_H   = 80;
const Y_R2   = Y_R1 + R1_H + 22; // 576 — Row 2: result boxes
const R2_H   = 88;
const H      = Y_R2 + R2_H + 20; // 684

function Arrowhead({ id, color }: { id: string; color: string }) {
  return (
    <marker id={id} markerWidth="9" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 9 3.5, 0 7" fill={color} />
    </marker>
  );
}

function CenterBox({
  y, h = BH, label, n, pct, sub, highlight = false,
}: {
  y: number; h?: number; label: string; n: string; pct?: string; sub?: string; highlight?: boolean;
}) {
  const hasSub  = !!sub;
  const hasPct  = !!pct && !hasSub;
  const yLabel  = y + h * (hasSub ? 0.22 : hasPct ? 0.26 : 0.35);
  const yN      = y + h * (hasSub ? 0.45 : hasPct ? 0.52 : 0.65);
  const yExtra  = y + h * (hasSub ? 0.70 : 0.78);
  return (
    <g>
      <rect x={CX} y={y} width={CW} height={h} rx={5}
        fill={highlight ? '#f0f7f2' : BOX_BG}
        stroke={GREEN} strokeWidth={highlight ? 2 : 1.5}
      />
      <text x={CM} y={yLabel} textAnchor="middle"
        fontSize={10} fill="#555" fontFamily="Arial, sans-serif">
        {label}
      </text>
      <text x={CM} y={yN} textAnchor="middle"
        fontSize={14} fontWeight="700" fill={highlight ? GREEN : '#111'}
        fontFamily="Arial, sans-serif">
        {n}
      </text>
      {(hasPct || hasSub) && (
        <text x={CM} y={yExtra} textAnchor="middle"
          fontSize={9} fill={highlight ? '#4a8c62' : '#888'} fontFamily="Arial, sans-serif">
          {pct ?? sub}
        </text>
      )}
    </g>
  );
}

function ExclusionBox({ y, label, n, pct }: { y: number; label: string; n: string; pct?: string }) {
  const mx = EX + EW / 2;
  return (
    <g>
      <rect x={EX} y={y} width={EW} height={BH} rx={4}
        fill={EXC_BG} stroke={EXC_BDR} strokeWidth={1} />
      <text x={mx} y={y + BH * (pct ? 0.26 : 0.36)} textAnchor="middle"
        fontSize={10} fill={EXC_TEXT} fontFamily="Arial, sans-serif">
        {label}
      </text>
      <text x={mx} y={y + BH * (pct ? 0.54 : 0.68)} textAnchor="middle"
        fontSize={13} fontWeight="700" fill={EXC_TEXT} fontFamily="Arial, sans-serif">
        {n}
      </text>
      {pct && (
        <text x={mx} y={y + BH * 0.82} textAnchor="middle"
          fontSize={9} fill="#e57373" fontFamily="Arial, sans-serif">
          {pct}
        </text>
      )}
    </g>
  );
}

function PhaseLabel({ label, y1, y2 }: { label: string; y1: number; y2: number }) {
  const mid = (y1 + y2) / 2;
  return (
    <g>
      <rect x={2} y={y1 + 2} width={PX - 6} height={y2 - y1 - 4} rx={4} fill={MID_GRN} />
      <text
        x={(PX - 6) / 2 + 2} y={mid}
        textAnchor="middle" dominantBaseline="middle"
        transform={`rotate(-90, ${(PX - 6) / 2 + 2}, ${mid})`}
        fontSize={9.5} fontWeight="700" fontFamily="Arial, sans-serif"
        fill={GREEN} letterSpacing="1.5"
      >
        {label}
      </text>
    </g>
  );
}

function DownArrow({ y }: { y: number }) {
  return (
    <line x1={CM} y1={y} x2={CM} y2={y + 16}
      stroke={GREEN} strokeWidth={1.5} markerEnd="url(#arr-down)" />
  );
}

function RightArrow({ sourceY }: { sourceY: number }) {
  const midY = sourceY + BH / 2;
  return (
    <line x1={CX + CW} y1={midY} x2={EX - 1} y2={midY}
      stroke={GREY_ARR} strokeWidth={1.5} strokeDasharray="5 3"
      markerEnd="url(#arr-right)" />
  );
}

export default function PrismaFlow({ className = '' }: { className?: string }) {
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      style={{ maxHeight: H, display: 'block' }}
      className={className}
      aria-label="PRISMA flow diagram"
    >
      <defs>
        <Arrowhead id="arr-down"  color={GREEN} />
        <Arrowhead id="arr-right" color={GREY_ARR} />
        <Arrowhead id="arr-hum"   color={AMBER} />
        <Arrowhead id="arr-llm"   color={SUPP_BDR} />
      </defs>

      {/* Phase background bands */}
      <rect x={0} y={0} width={W} height={P1} fill={LIGHT_GRN} />
      <rect x={0} y={P1} width={W} height={P2 - P1} fill="#fafafa" />
      <rect x={0} y={P2} width={W} height={P3 - P2} fill={LIGHT_GRN} />
      <rect x={0} y={P3} width={W} height={H - P3} fill="#fafafa" />

      {/* Phase labels */}
      <PhaseLabel label="IDENTIFICATION" y1={0}  y2={P1} />
      <PhaseLabel label="SCREENING"      y1={P1} y2={P2} />
      <PhaseLabel label="ELIGIBILITY"    y1={P2} y2={P3} />
      <PhaseLabel label="INCLUDED"       y1={P3} y2={H}  />

      {/* Box 1 — Records identified */}
      <CenterBox y={Y1} h={BH1}
        label="Records identified via database searching"
        n="n = 40,653"
        sub="Scopus 17,021 · WoS 15,179 · CAB 5,723 · ASP 1,187 · 24 other sources 1,543"
      />
      <DownArrow y={Y1 + BH1} />

      {/* Box 2 — After deduplication */}
      <CenterBox y={Y2} label="Records after deduplication" n="n = 26,182" pct="64% of all identified" />
      <ExclusionBox y={Y2} label="Duplicates removed" n="n = 14,471" pct="36% of all identified" />
      <RightArrow sourceY={Y2} />
      <DownArrow y={Y2 + BH} />

      {/* Box 3 — Screened */}
      <CenterBox y={Y3} label="Records screened (title & abstract)" n="n = 26,182" />
      <ExclusionBox y={Y3} label="Records excluded" n="n = 17,429" pct="67% of screened" />
      <RightArrow sourceY={Y3} />
      <DownArrow y={Y3 + BH} />

      {/* Box 4 — Full texts sought */}
      <CenterBox y={Y4} label="Full texts sought for retrieval" n="n = 8,748" pct="33% of screened" />
      <ExclusionBox y={Y4} label="Not retrieved" n="n = 5,243" pct="60% sought" />
      <RightArrow sourceY={Y4} />
      <DownArrow y={Y4 + BH} />

      {/* Box 5 — Full texts retrieved (auto-retrieved + manual procurement) */}
      <CenterBox y={Y5}
        label="Full texts retrieved (auto-retrieved + manual procurement)"
        n="n = 3,505"
        pct="40% of full texts sought"
      />
      <DownArrow y={Y5 + BH} />

      {/* ── INCLUDED: two-track fork ─────────────────────────────── */}

      {/* Vertical connector from arrow end to fork bar */}
      <line x1={CM} y1={Y5 + BH + 16} x2={CM} y2={Y_FORK}
        stroke={GREEN} strokeWidth={1.5} />

      {/* Horizontal fork bar spanning both track centres */}
      <line x1={LLM_C} y1={Y_FORK} x2={HUM_C} y2={Y_FORK}
        stroke={GREEN} strokeWidth={1.5} />

      {/* Arrows down to Row 1 boxes */}
      <line x1={LLM_C} y1={Y_FORK} x2={LLM_C} y2={Y_R1 - 1}
        stroke={SUPP_BDR} strokeWidth={1.5} markerEnd="url(#arr-llm)" />
      <line x1={HUM_C} y1={Y_FORK} x2={HUM_C} y2={Y_R1 - 1}
        stroke={AMBER_BDR} strokeWidth={1.5} markerEnd="url(#arr-hum)" />

      {/* Row 1 — LLM track overview */}
      <rect x={LLM_X} y={Y_R1} width={TW} height={R1_H} rx={5}
        fill={SUPP_BG} stroke={SUPP_BDR} strokeWidth={1.5} />
      <text x={LLM_C} y={Y_R1 + R1_H * 0.18} textAnchor="middle"
        fontSize={9} fill="#475569" fontWeight="700" fontFamily="Arial, sans-serif" letterSpacing="0.5">
        LLM SCREENING
      </text>
      <text x={LLM_C} y={Y_R1 + R1_H * 0.38} textAnchor="middle"
        fontSize={9.5} fill="#555" fontFamily="Arial, sans-serif">
        n = 3,505 assessed
      </text>
      <text x={LLM_C} y={Y_R1 + R1_H * 0.57} textAnchor="middle"
        fontSize={9.5} fill={EXC_TEXT} fontFamily="Arial, sans-serif">
        n = 1,096 excluded (31%)
      </text>
      <text x={LLM_C} y={Y_R1 + R1_H * 0.80} textAnchor="middle"
        fontSize={8.5} fill="#94a3b8" fontFamily="Arial, sans-serif">
        auto-retrieved corpus only
      </text>

      {/* Row 1 — Human track overview */}
      <rect x={HUM_X} y={Y_R1} width={TW} height={R1_H} rx={5}
        fill={AMBER_BG} stroke={AMBER_BDR} strokeWidth={1.5} />
      <text x={HUM_C} y={Y_R1 + R1_H * 0.18} textAnchor="middle"
        fontSize={9} fill={AMBER} fontWeight="700" fontFamily="Arial, sans-serif" letterSpacing="0.5">
        HUMAN CODING
      </text>
      <text x={HUM_C} y={Y_R1 + R1_H * 0.38} textAnchor="middle"
        fontSize={9.5} fill="#555" fontFamily="Arial, sans-serif">
        5 rounds · 100 coded
      </text>
      <text x={HUM_C} y={Y_R1 + R1_H * 0.57} textAnchor="middle"
        fontSize={9.5} fill={EXC_TEXT} fontFamily="Arial, sans-serif">
        n = 14 excluded (14%)
      </text>
      <text x={HUM_C} y={Y_R1 + R1_H * 0.80} textAnchor="middle"
        fontSize={8.5} fill="#94a3b8" fontFamily="Arial, sans-serif">
        incl. manually procured PDFs
      </text>

      {/* Arrows Row 1 → Row 2 */}
      <line x1={LLM_C} y1={Y_R1 + R1_H} x2={LLM_C} y2={Y_R2 - 1}
        stroke={SUPP_BDR} strokeWidth={1.5} markerEnd="url(#arr-llm)" />
      <line x1={HUM_C} y1={Y_R1 + R1_H} x2={HUM_C} y2={Y_R2 - 1}
        stroke={AMBER_BDR} strokeWidth={1.5} markerEnd="url(#arr-hum)" />

      {/* Row 2 — LLM included */}
      <rect x={LLM_X} y={Y_R2} width={TW} height={R2_H} rx={5}
        fill={SUPP_BG} stroke={SUPP_BDR} strokeWidth={1.5} />
      <text x={LLM_C} y={Y_R2 + R2_H * 0.20} textAnchor="middle"
        fontSize={9} fill="#475569" fontWeight="700" fontFamily="Arial, sans-serif" letterSpacing="0.5">
        LLM INCLUDED
      </text>
      <text x={LLM_C} y={Y_R2 + R2_H * 0.50} textAnchor="middle"
        fontSize={16} fontWeight="700" fill="#64748b" fontFamily="Arial, sans-serif">
        n = 2,368
      </text>
      <text x={LLM_C} y={Y_R2 + R2_H * 0.72} textAnchor="middle"
        fontSize={9.5} fill="#64748b" fontFamily="Arial, sans-serif">
        68% of retrieved
      </text>
      <text x={LLM_C} y={Y_R2 + R2_H * 0.90} textAnchor="middle"
        fontSize={8.5} fill="#94a3b8" fontFamily="Arial, sans-serif">
        supplementary reference set
      </text>

      {/* Row 2 — Human included (PRIMARY) */}
      <rect x={HUM_X} y={Y_R2} width={TW} height={R2_H} rx={5}
        fill={AMBER_BG} stroke={AMBER_BDR} strokeWidth={2} />
      <text x={HUM_C} y={Y_R2 + R2_H * 0.18} textAnchor="middle"
        fontSize={9} fill={AMBER} fontWeight="700" fontFamily="Arial, sans-serif" letterSpacing="0.5">
        HUMAN INCLUDED ★
      </text>
      <text x={HUM_C} y={Y_R2 + R2_H * 0.48} textAnchor="middle"
        fontSize={16} fontWeight="700" fill={AMBER} fontFamily="Arial, sans-serif">
        n = 86
      </text>
      <text x={HUM_C} y={Y_R2 + R2_H * 0.70} textAnchor="middle"
        fontSize={9.5} fill={AMBER} fontFamily="Arial, sans-serif">
        86% of coded
      </text>
      <text x={HUM_C} y={Y_R2 + R2_H * 0.88} textAnchor="middle"
        fontSize={8.5} fill={AMBER} fontFamily="Arial, sans-serif">
        PRIMARY OUTPUT · ongoing
      </text>
    </svg>
  );
}
