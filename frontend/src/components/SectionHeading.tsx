'use client';

import { useState, useCallback } from 'react';
import { Link } from 'lucide-react';

interface SectionHeadingProps {
  id: string;
  children: React.ReactNode;
  className?: string;
}

export default function SectionHeading({ id, children, className = '' }: SectionHeadingProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    const url = `${window.location.origin}${window.location.pathname}#${id}`;
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  }, [id]);

  return (
    <h2 className={`group/heading flex items-center gap-2 ${className}`}>
      {children}
      <button
        onClick={handleCopy}
        title="Copy link to section"
        className="relative opacity-0 group-hover/heading:opacity-100 transition-opacity duration-150 text-gray-400 hover:text-green focus:outline-none"
        aria-label="Copy link to section"
      >
        <Link className="w-3 h-3" />
        {copied && (
          <span className="absolute left-6 top-1/2 -translate-y-1/2 text-xs font-tagline font-semibold text-green bg-white border border-green/20 rounded px-2 py-0.5 whitespace-nowrap shadow-sm pointer-events-none">
            Copied!
          </span>
        )}
      </button>
    </h2>
  );
}
