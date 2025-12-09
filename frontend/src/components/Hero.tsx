"use client";

import { motion } from "framer-motion";
import { HiChevronDown } from "react-icons/hi";

export default function Hero() {
  return (
    <div className="relative h-screen w-full text-white overflow-hidden">
      
      {/* 
        ORIGINAL VIDEO SOURCE (for attribution records only):
        https://www.pexels.com/video/drone-footage-of-rice-field-terraces-in-a-valley-2994205/
      */}

      {/* Background Video */}
      <video
        autoPlay
        muted
        loop
        playsInline
        className="absolute top-0 left-0 w-full h-full object-cover z-[-1]"
      >
        <source
          src="https://bristlepine.s3.us-east-2.amazonaws.com/2994205-uhd_3840_2160_30fps_clip.mp4"
          type="video/webm"
        />
      </video>

      {/* Overlay */}
      <div className="absolute inset-0 bg-black/60 z-0" />

      {/* HERO CONTENT ONLY — no other sections */}
      <div className="relative z-10 h-full flex flex-col justify-center items-center text-center px-6">

        {/* Extra top spacing */}
        <div className="mt-24 md:mt-32" />

        {/* Project Title */}
        <motion.h1
          className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-logo text-sand max-w-4xl leading-tight whitespace-pre-line"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          {"Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector."}
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          className="text-sand/90 text-base md:text-lg font-tagline mt-10"
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
        >
          Project sponsored by the International Livestock Research Institute (ILRI)
        </motion.p>

        {/* Citation */}
        <motion.p
          className="max-w-3xl text-xs md:text-sm mt-12 text-clay font-tagline italic tracking-wide"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.35 }}
        >
          Cissé, J. D., Staub, C. G., & Khan, Z. (2025). Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector. https://doi.org/10.5281/zenodo.17809739
        </motion.p>

        {/* Additional spacing before arrow */}
        <div className="mt-16" />

        {/* ENTER ARROW → now goes to /about */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.6 }}
        >
          <a
            href="/about"
            className="inline-flex flex-col items-center text-sand hover:text-white transition-colors group"
          >
            <span className="font-tagline text-sm tracking-wider">ENTER</span>
            <motion.div
              animate={{ y: [0, 6, 0] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            >
              <HiChevronDown className="h-8 w-8" />
            </motion.div>
          </a>
        </motion.div>

      </div>
    </div>
  );
}
