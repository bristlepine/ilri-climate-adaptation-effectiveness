'use client';

import { useEffect, useState } from 'react';
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

interface Props {
  src: string;            // /map/data/xxx.json  — Plotly figure JSON
  fallbackImg?: string;   // /map/xxx.png         — shown if JSON unavailable
  pngSrc?: string;        // /map/xxx.png         — for "Download PNG" button
  csvSrc?: string;        // /map/data/xxx.csv    — for "Download CSV" button
  height?: number;
  className?: string;
  config?: object;
}

export default function PlotlyChart({ src, fallbackImg, pngSrc, csvSrc, height = 400, className = '', config }: Props) {
  const [figData, setFigData] = useState<any>(null);
  const [error, setError]     = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setError(false);
    fetch(src)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(d => { setFigData(d); setLoading(false); })
      .catch(() => { setError(true); setLoading(false); });
  }, [src]);

  const mergedConfig = {
    responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: { format: 'png', scale: 2 },
    ...(config ?? {}),
  };

  const dlBtn = (href: string, ext: string) => (
    <a
      href={href}
      download
      onClick={e => e.stopPropagation()}
      className="flex items-center gap-1 text-xs font-tagline font-semibold px-2.5 py-1.5 rounded-full border border-gray-200 text-gray-500 hover:border-green hover:text-green transition bg-white/80 backdrop-blur-sm"
    >
      <Download className="w-3 h-3" />
      {ext}
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

  return (
    <div className={`relative ${className}`} onClick={e => e.stopPropagation()}>
      <Plot
        data={figData.data ?? []}
        layout={{ ...(figData.layout ?? {}), autosize: true }}
        config={mergedConfig as any}
        style={{ width: '100%', height }}
        useResizeHandler
      />
      {(pngSrc || csvSrc) && (
        <div className="flex items-center justify-end gap-1.5 px-3 py-1.5 border-t border-gray-100 bg-gray-50/60">
          {pngSrc && dlBtn(pngSrc, 'PNG')}
          {csvSrc && dlBtn(csvSrc, 'CSV')}
        </div>
      )}
    </div>
  );
}
