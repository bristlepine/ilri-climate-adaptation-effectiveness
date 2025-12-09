'use client';

import { useState } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { FaLinkedin } from 'react-icons/fa';
import { HiOutlineMail } from 'react-icons/hi';
import { IoDocumentTextOutline } from 'react-icons/io5';
import { teamMembers } from '@/lib/data';

export default function TeamPage() {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <main className="page-wrapper relative">
      <Navbar />

      <section className="px-6 mt-16 mb-16 py-24 text-center">
        <h1 className="text-4xl font-logo text-green mb-6">Meet the Team</h1>
        <p className="max-w-2xl mx-auto text-lg mb-12">
          Get to know the passionate individuals behind Bristlepine.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-12 max-w-6xl mx-auto">
          {teamMembers.map((member, index) => (
            <motion.div
              key={index}
              className="group"
              whileHover={{ scale: 1.05 }}
            >
              {/* Wrapper for avatar and tooltip */}
              <div
                className="relative w-32 h-32 mx-auto mb-4"
                onMouseEnter={() => setHoveredIndex(index)}
                onMouseLeave={() => setHoveredIndex(null)}
              >
                <Link href={`/team/${member.slug}`} aria-label={`View profile of ${member.name}`}>
                    <div className="w-full h-full rounded-full overflow-hidden border-2 border-green shadow-md cursor-pointer">
                        <img
                        src={member.image}
                        alt={member.name}
                        className="w-full h-full object-cover"
                        />
                    </div>
                </Link>
                
                {/* Tooltip for shortBio */}
                <AnimatePresence>
                  {hoveredIndex === index && (
                    <motion.div
                      className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 w-64 p-4 bg-charcoal text-white text-center rounded-lg shadow-xl z-20"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 10, transition: { duration: 0.15 } }}
                      transition={{ type: 'spring', stiffness: 300, damping: 20, duration: 0.2 }}
                    >
                      <p className="text-sm font-medium">{member.shortBio}</p>
                      <p className="text-xs text-stone-300 italic mt-2">Click for full bio</p>
                      
                      <div className="absolute top-full left-1/2 -translate-x-1/2 border-x-8 border-x-transparent border-t-8 border-t-charcoal"></div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <h3 className="text-lg font-tagline font-semibold text-green">{member.name}</h3>
              <p className="text-sm font-tagline text-charcoal">{member.role}</p>
              <div className="flex justify-center gap-4 mt-2 text-green text-xl">
                <a href={member.linkedin} target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                  <FaLinkedin className="hover:opacity-70" />
                </a>
                <a href={`mailto:${member.email}`} aria-label="Email">
                  <HiOutlineMail className="hover:opacity-70" />
                </a>
                <a href={member.cv} target="_blank" rel="noopener noreferrer" aria-label="CV">
                  <IoDocumentTextOutline className="hover:opacity-70" />
                </a>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      <Footer />
    </main>
  );
}