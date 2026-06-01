'use client';

import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { Download } from 'lucide-react';

const Plot = dynamic(
  () =>
    import('react-plotly.js/factory').then(async (factory) => {
      const Plotly = await import('plotly.js-dist-min');
      return factory.default(Plotly as any);
    }),
  { ssr: false, loading: () => <div className="animate-pulse bg-gray-100 rounded-xl w-full h-full min-h-[200px]" /> }
);

interface Tooltip { text: string; x: number; y: number }

interface Props {
  src: string;
  fallbackImg?: string;
  pngSrc?: string;
  csvSrc?: string;
  height?: number;
  className?: string;
  config?: object;
  patchFigure?: (fig: any) => any;
  /** y-axis label → definition; shown as HTML tooltip when hovering the label text */
  yAxisTooltips?: Record<string, string>;
}

export default function PlotlyChart({ src, fallbackImg, pngSrc, csvSrc, height = 400, className = '', config, patchFigure, yAxisTooltips }: Props) {
  const [figData, setFigData]   = useState<any>(null);
  const [error, setError]       = useState(false);
  const [loading, setLoading]   = useState(true);
  const [tooltip, setTooltip]   = useState<Tooltip | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const cleanupRef   = useRef<(() => void) | null>(null);

  useEffect(() => {
    setLoading(true); setError(false);
    fetch(src)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(d => { setFigData(d); setLoading(false); })
      .catch(() => { setError(true); setLoading(false); });
  }, [src]);

  // Attach mouseenter/mouseleave to every .ytick text SVG node that has a definition.
  // This fires after every Plotly render via onAfterPlot.
  const attachYTooltips = () => {
    if (!yAxisTooltips || !containerRef.current) return;

    // Clean up previous listeners to avoid duplicates on re-render
    cleanupRef.current?.();
    const removers: (() => void)[] = [];

    const nodes = containerRef.current.querySelectorAll<SVGTextElement>(
      '.ytick text, g.ytick text'
    );
    nodes.forEach(el => {
      const label = el.textContent?.trim() ?? '';
      const def   = yAxisTooltips[label];
      if (!def) return;

      el.style.cursor = 'help';

      const onEnter = () => {
        if (!containerRef.current) return;
        const er = el.getBoundingClientRect();
        const cr = containerRef.current.getBoundingClientRect();
        setTooltip({ text: def, x: er.right - cr.left + 8, y: er.top - cr.top });
      };
      const onLeave = () => setTooltip(null);

      el.addEventListener('mouseenter', onEnter);
      el.addEventListener('mouseleave', onLeave);
      removers.push(() => { el.removeEventListener('mouseenter', onEnter); el.removeEventListener('mouseleave', onLeave); });
    });

    cleanupRef.current = () => removers.forEach(r => r());
  };

  // Also clean up on unmount
  useEffect(() => () => { cleanupRef.current?.(); }, []);

  const mergedConfig = {
    responsive: true, displaylogo: false, scrollZoom: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: { format: 'png', scale: 2 },
    ...(config ?? {}),
  };

  const dlBtn = (href: string, ext: string) => (
    <a href={href} download onClick={e => e.stopPropagation()}
      className="flex items-center gap-1 text-xs font-tagline font-semibold px-2.5 py-1.5 rounded-full border border-gray-200 text-gray-500 hover:border-green hover:text-green transition bg-white/80 backdrop-blur-sm">
      <Download className="w-3 h-3" />{ext}
    </a>
  );

  if (loading) return (
    <div className={`animate-pulse bg-gray-50 flex items-center justify-center ${className}`} style={{ height }}>
      <span className="text-xs font-tagline text-gray-400">Loading chart…</span>
    </div>
  );

  if (error || !figData) {
    if (fallbackImg) return <img src={fallbackImg} alt="Chart" className={`w-full object-contain ${className}`} style={{ height }} />;
    return <div className={`flex items-center justify-center bg-gray-50 ${className}`} style={{ height }}><span className="text-xs text-gray-400">Chart unavailable</span></div>;
  }

  const fig = patchFigure ? patchFigure(figData) : figData;

  return (
    <div ref={containerRef} className={`relative ${className}`} onClick={e => e.stopPropagation()}>
      <Plot
        data={fig.data ?? []}
        layout={{ ...(fig.layout ?? {}), autosize: true }}
        config={mergedConfig as any}
        style={{ width: '100%', height }}
        useResizeHandler
        onAfterPlot={attachYTooltips}
      />

      {/* HTML tooltip anchored to the hovered y-axis label */}
      {tooltip && (
        <div
          className="pointer-events-none absolute z-50 max-w-[220px] rounded-lg border border-gray-200 bg-white px-3 py-2 shadow-lg text-xs font-tagline text-gray-700 leading-relaxed"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          {tooltip.text}
        </div>
      )}

      {(pngSrc || csvSrc) && (
        <div className="flex items-center justify-end gap-1.5 px-3 py-1.5 border-t border-gray-100 bg-gray-50/60">
          {pngSrc && dlBtn(pngSrc, 'PNG')}
          {csvSrc && dlBtn(csvSrc, 'CSV')}
        </div>
      )}
    </div>
  );
}
