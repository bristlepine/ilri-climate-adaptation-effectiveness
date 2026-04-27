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

// Layout constants
const W   = 800;
const H   = 550;
const PX  = 92;    // phase label band width
const CX  = PX + 8;  // center column left edge
const CW  = 320;     // center column width
const CM  = CX + CW / 2;  // center column midpoint
const EX  = CX + CW + 28; // exclusion column left edge
const EW  = W - EX - 6;   // exclusion column width
const BH  = 52;   // standard box height
const BH1 = 80;   // identification box height

// Box y-positions
const Y1 = 14;
const Y2 = Y1 + BH1 + 22;       // 116
const P1 = Y2 + BH + 18;        // end of Identification: 186
const Y3 = P1 + 14;             // 200
const P2 = Y3 + BH + 18;        // end of Screening: 270
const Y4 = P2 + 14;             // 284
const Y5 = Y4 + BH + 22;        // 358
const P3 = Y5 + BH + 18;        // end of Eligibility: 428
const Y6 = P3 + 16;             // 444

function Arrowhead({ id, color }: { id: string; color: string }) {
  return (
    <marker id={id} markerWidth="9" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 9 3.5, 0 7" fill={color} />
    </marker>
  );
}

function CenterBox({
  y, h = BH, label, n, sub, highlight = false,
}: {
  y: number; h?: number; label: string; n: string; sub?: string; highlight?: boolean;
}) {
  return (
    <g>
      <rect x={CX} y={y} width={CW} height={h} rx={5}
        fill={highlight ? '#f0f7f2' : BOX_BG}
        stroke={GREEN} strokeWidth={highlight ? 2 : 1.5}
      />
      <text x={CM} y={y + (sub ? h * 0.28 : h * 0.35)} textAnchor="middle"
        fontSize={10} fill="#555" fontFamily="Arial, sans-serif">
        {label}
      </text>
      <text x={CM} y={y + (sub ? h * 0.52 : h * 0.65)} textAnchor="middle"
        fontSize={14} fontWeight="700" fill={highlight ? GREEN : '#111'}
        fontFamily="Arial, sans-serif">
        {n}
      </text>
      {sub && (
        <text x={CM} y={y + h * 0.78} textAnchor="middle"
          fontSize={8.5} fill="#888" fontFamily="Arial, sans-serif">
          {sub}
        </text>
      )}
    </g>
  );
}

function ExclusionBox({ y, label, n }: { y: number; label: string; n: string }) {
  return (
    <g>
      <rect x={EX} y={y} width={EW} height={BH} rx={4}
        fill={EXC_BG} stroke={EXC_BDR} strokeWidth={1} />
      <text x={EX + EW / 2} y={y + BH * 0.36} textAnchor="middle"
        fontSize={10} fill={EXC_TEXT} fontFamily="Arial, sans-serif">
        {label}
      </text>
      <text x={EX + EW / 2} y={y + BH * 0.68} textAnchor="middle"
        fontSize={13} fontWeight="700" fill={EXC_TEXT} fontFamily="Arial, sans-serif">
        {n}
      </text>
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
        <Arrowhead id="arr-down" color={GREEN} />
        <Arrowhead id="arr-right" color={GREY_ARR} />
      </defs>

      {/* Phase background bands */}
      <rect x={0} y={0} width={W} height={P1} fill={LIGHT_GRN} />
      <rect x={0} y={P1} width={W} height={P2 - P1} fill="#fafafa" />
      <rect x={0} y={P2} width={W} height={P3 - P2} fill={LIGHT_GRN} />
      <rect x={0} y={P3} width={W} height={H - P3} fill="#fafafa" />

      {/* Phase labels */}
      <PhaseLabel label="IDENTIFICATION" y1={0} y2={P1} />
      <PhaseLabel label="SCREENING" y1={P1} y2={P2} />
      <PhaseLabel label="ELIGIBILITY" y1={P2} y2={P3} />
      <PhaseLabel label="INCLUDED" y1={P3} y2={H} />

      {/* Box 1 — Records identified */}
      <CenterBox y={Y1} h={BH1}
        label="Records identified via database searching"
        n="n = 39,113"
        sub="Scopus 17,021 · WoS 15,179 · CAB 5,723 · ASP 1,187 · AGRIS 3"
      />

      {/* Arrow 1 → 2 */}
      <DownArrow y={Y1 + BH1} />

      {/* Box 2 — After deduplication */}
      <CenterBox y={Y2} label="Records after deduplication" n="n = 25,208" />
      <ExclusionBox y={Y2} label="Duplicates removed" n="n = 13,905" />
      <RightArrow sourceY={Y2} />

      {/* Arrow 2 → 3 */}
      <DownArrow y={Y2 + BH} />

      {/* Box 3 — Screened */}
      <CenterBox y={Y3} label="Records screened (title & abstract)" n="n = 25,208" />
      <ExclusionBox y={Y3} label="Records excluded" n="n = 16,653" />
      <RightArrow sourceY={Y3} />

      {/* Arrow 3 → 4 */}
      <DownArrow y={Y3 + BH} />

      {/* Box 4 — Full texts sought */}
      <CenterBox y={Y4} label="Full texts sought for retrieval" n="n = 8,555" />
      <ExclusionBox y={Y4} label="Not retrieved" n="n = 5,079" />
      <RightArrow sourceY={Y4} />

      {/* Arrow 4 → 5 */}
      <DownArrow y={Y4 + BH} />

      {/* Box 5 — Full texts assessed */}
      <CenterBox y={Y5} label="Full texts assessed for eligibility" n="n = 3,476" />
      <ExclusionBox y={Y5} label="Full texts excluded" n="n = 726" />
      <RightArrow sourceY={Y5} />

      {/* Arrow 5 → 6 */}
      <DownArrow y={Y5 + BH} />

      {/* Box 6 — Included */}
      <CenterBox y={Y6} h={BH + 8}
        label="Studies included in data extraction"
        n="n = 2,750" highlight
      />
    </svg>
  );
}
