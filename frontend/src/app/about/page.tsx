'use client';

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { motion } from "framer-motion";

export default function AboutPage() {
    return (
        <main className="page-wrapper">
            <Navbar />

            {/* Hero Section */}
            <section className="relative h-[80vh] w-full text-white overflow-hidden">
                <video
                    autoPlay
                    muted
                    loop
                    playsInline
                    className="absolute inset-0 w-full h-full object-cover z-[-1]"
                >
                    <source
                        src="https://bristlepine.s3.us-east-2.amazonaws.com/hero.webm"
                        type="video/webm"
                    />
                </video>

                <div className="absolute inset-0 bg-black/60 z-0" />

                <motion.div
                    className="relative z-10 flex flex-col justify-center items-center text-center px-6 pt-20 h-full"
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1 }}
                >
                    <h1 className="text-5xl font-logo text-sand mb-4">About Bristlepine</h1>
                    <p className="text-lg font-tagline text-white/90 max-w-2xl mx-auto">
                        Building climate resilience through informed, inclusive, and systems-based solutions.
                    </p>
                </motion.div>
            </section>

            {/* Mission + Vision Side by Side */}
            <section className="px-6 py-24 bg-white text-charcoal">
                <div className="max-w-6xl mx-auto grid md:grid-cols-2 gap-12 items-start">
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                        viewport={{ once: true }}
                    >
                        <h2 className="text-3xl font-logo text-green mb-4">Our Vision</h2>
                        <p className="text-lg font-tagline">
                            A healthy and climate-resilient world where communities and institutions adapt and thrive, driven by informed, inclusive, and proactive decision-making.
                        </p>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                        viewport={{ once: true }}
                    >
                        <h2 className="text-3xl font-logo text-green mb-4">Our Mission</h2>
                        <p className="text-lg font-tagline">
                            We empower governments, communities, institutions, and development partners to make informed decisions about climate-related risks and opportunities to build resilience, promote sustainable development, and create positive social impact.
                        </p>
                    </motion.div>
                </div>
            </section>

            {/* Philosophy Section */}
            <section className="px-6 py-24 bg-sage/10 text-center text-charcoal">
                <motion.div
                    className="max-w-3xl mx-auto"
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    viewport={{ once: true }}
                >
                    <h2 className="text-3xl font-logo text-green mb-6">Our Philosophy</h2>
                    <p className="text-lg font-tagline">
                        At Bristlepine, we approach resilience as a systems challenge — one that demands science, collaboration, equity, and innovation. We partner with stakeholders at every level to build strategies that are just, actionable, and rooted in local context.
                    </p>
                </motion.div>
            </section>

            {/* Why Bristlepine – Video Background (Parallax Style) */}
            <section className="relative h-[90vh] text-sand overflow-hidden flex items-center justify-center text-center px-6">
                <video
                    autoPlay
                    muted
                    loop
                    playsInline
                    className="absolute inset-0 w-full h-full object-cover z-[-1]"
                >
                    <source
                        src="https://bristlepine.s3.us-east-2.amazonaws.com/hero.webm"
                        type="video/webm"
                    />
                </video>
                <div className="absolute inset-0 bg-black/60 z-0" />
                <motion.div
                    className="relative z-10 max-w-3xl"
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1 }}
                    viewport={{ once: true }}
                >
                    <h2 className="text-3xl font-logo mb-4">Why “Bristlepine”?</h2>
                    <p className="text-lg font-tagline text-white/90 mb-4">
                        The name is inspired by the <strong>Bristlecone Pine</strong> — one of the oldest living organisms on Earth. Thriving in harsh, high-elevation conditions, these trees grow slowly, adapt continually, and endure for thousands of years.
                    </p>
                    <p className="text-lg font-tagline text-white/90">
                        Like the bristlecone, we believe resilience is about more than surviving — it’s about adapting with wisdom, investing in longevity, and growing stronger through challenge.
                    </p>
                </motion.div>
            </section>

            <Footer />
        </main>
    );
}
