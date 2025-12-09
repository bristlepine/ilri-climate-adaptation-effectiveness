'use client';

import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';

export default function DataPage() {
  return (
    <main className="page-wrapper min-h-screen flex flex-col">
      <Navbar />

      {/* Placeholder content */}
      <section className="max-w-4xl mx-auto px-6 py-24 flex-grow">
        <h1 className="text-3xl font-logo text-green mb-6">Data</h1>
        <p className="text-charcoal font-tagline">
          This page will include the evidence table, filters, data dictionary, and downloads.
        </p>
      </section>

      <Footer />
    </main>
  );
}
