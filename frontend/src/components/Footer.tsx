'use client';

import { FaLinkedin, FaEnvelope } from "react-icons/fa";

export default function Footer() {
  return (
    <footer className="bg-charcoal text-clay py-12 px-6"> {/* reduced vertical padding */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12">
        {/* Logo + Tagline */}
        <div className="flex flex-col space-y-4">
          <div className="flex items-center gap-3">
            <img
              src="/logo_tree_sand.svg"
              alt="Bristlepine Logo"
              className="h-8 w-auto relative top-[-3px]"
            />
            <span className="text-xl font-logo font-bold text-sand">
              Bristlepine
            </span>
          </div>
          <p className="text-sm leading-relaxed max-w-xs font-tagline text-clay">
            Empowering climate resilience through intelligence, innovation, and
            impact.
          </p>
        </div>

        {/* Contact Info */}
        <div>
          <h4 className="text-lg font-semibold mb-4 text-sand">Contact</h4>
          <div className="flex items-center gap-2 mb-2">
            <FaEnvelope className="text-sand text-sm" />
            <a
              href="mailto:info@bristlep.com"
              className="text-sm font-tagline hover:text-sage transition"
            >
              info@bristlep.com
            </a>
          </div>
          <div className="flex items-center gap-2">
            <FaLinkedin className="text-sand text-sm" />
            <a
              href="https://www.linkedin.com/company/bristlepine"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-tagline hover:text-sage transition"
            >
              Follow us on LinkedIn
            </a>
          </div>
        </div>
      </div>

      {/* Bottom Line */}
      <div className="mt-10 border-t border-[#3A3A3A] pt-6 text-center text-sm text-[#A0A0A0] font-tagline">
        &copy; {new Date().getFullYear()} Bristlepine LLC. All rights reserved.
      </div>
    </footer>
  );
}
