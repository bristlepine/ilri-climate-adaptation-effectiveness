'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X } from 'lucide-react';
import { FaGithub } from "react-icons/fa";
import { usePathname } from "next/navigation";

const links: { name: string; href: string }[] = [
  { name: 'Home', href: '/' },
  { name: 'About', href: '/about' },
  { name: 'Protocols', href: '/protocols' },
  { name: 'Evidence', href: '/evidence' },
  { name: 'Review', href: '/review' },
  { name: 'Data', href: '/data' }
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const pathname = usePathname();
  const isHomePage = pathname === "/";

  useEffect(() => {
    if (!isHomePage) {
      setScrolled(true);
      return;
    }

    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const textColor = scrolled ? 'text-green' : 'text-sand';
  const hoverColor = scrolled ? 'hover:text-sage' : 'hover:text-clay';
  const logoSrc = scrolled ? '/logo_tree_green.svg' : '/logo_tree_sand.svg';

  return (
    <motion.nav
      className={`fixed top-0 w-full z-50 px-6 py-4 transition-all duration-300 ${
        scrolled ? 'bg-white/80 backdrop-blur shadow' : 'bg-transparent'
      }`}
      initial={isHomePage ? { y: -100 } : false}
      animate={isHomePage ? { y: 0 } : false}
      transition={isHomePage ? { duration: 0.5 } : {}}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between">

        {/* BRAND LOGO -> Bristlepine home page */}
        <a
          href="https://bristlepineconsulting.com"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3"
        >
          <img
            src={logoSrc}
            alt="Bristlepine Logo"
            className="h-6 w-auto transition-opacity duration-300"
          />
          <span
            className={`text-2xl font-logo font-bold tracking-tight transition-colors duration-300 ${textColor}`}
          >
            Bristlepine
          </span>
        </a>

        {/* DESKTOP NAV */}
        <div
          className={`hidden md:flex items-center gap-6 text-sm font-tagline font-medium transition-colors duration-300 ${textColor}`}
        >
          {links.map(({ name, href }) => (
            <Link
              key={name}
              href={href}
              className={`transition duration-200 ${hoverColor}`}
            >
              {name}
            </Link>
          ))}

          {/* GitHub icon – aligned cleanly */}
          <a
            href="https://github.com/bristlepine/ilri-climate-adaptation-effectiveness"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center"
          >
            <FaGithub
              className={`w-5 h-5 relative top-[1px] ${textColor} ${hoverColor}`}
            />
          </a>
        </div>

        {/* MOBILE MENU ICON */}
        <div className="md:hidden">
          <button onClick={() => setMenuOpen(!menuOpen)}>
            {menuOpen ? (
              <X className={`w-6 h-6 ${textColor}`} />
            ) : (
              <Menu className={`w-6 h-6 ${textColor}`} />
            )}
          </button>
        </div>
      </div>

      {/* MOBILE DROPDOWN */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            className={`md:hidden mt-4 px-6 pb-4 rounded shadow backdrop-blur transition-colors duration-300 ${
              scrolled ? 'bg-sand/95 text-charcoal' : 'bg-black/80 text-sand'
            }`}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            <div className="flex flex-col space-y-3 font-tagline font-medium">
              {links.map(({ name, href }) => (
                <Link
                  key={name}
                  href={href}
                  className={`transition ${
                    scrolled ? 'hover:text-green' : 'hover:text-clay'
                  }`}
                  onClick={() => setMenuOpen(false)}
                >
                  {name}
                </Link>
              ))}

              {/* GitHub – icon only */}
              <a
                href="https://github.com/bristlepine/ilri-climate-adaptation-effectiveness"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center"
                onClick={() => setMenuOpen(false)}
              >
                <FaGithub className="w-6 h-6 relative top-[1px]" />
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
}
