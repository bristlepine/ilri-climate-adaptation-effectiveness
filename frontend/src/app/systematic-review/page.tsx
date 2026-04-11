import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export default function SystematicReviewPage() {
  return (
    <main className="page-wrapper min-h-screen flex flex-col">
      <Navbar />

      {/* Hero */}
      <section className="relative min-h-[40vh] flex items-end px-6 pt-32 pb-16 text-sand overflow-hidden">
        <video autoPlay muted loop playsInline className="absolute inset-0 w-full h-full object-cover z-[-1]">
          <source src="https://bristlepine.s3.us-east-2.amazonaws.com/2994205-uhd_3840_2160_30fps_clip.mp4" type="video/mp4" />
        </video>
        <div className="absolute inset-0 bg-black/65 z-0" />
        <div className="relative z-10 max-w-5xl mx-auto w-full">
          <p className="text-sm font-tagline uppercase tracking-widest text-clay mb-3">Deliverables D6 – D9</p>
          <h1 className="text-4xl md:text-5xl font-logo font-bold mb-4">Systematic Review & Meta-Analysis</h1>
          <p className="text-lg font-tagline text-clay max-w-2xl">
            A systematic review and meta-analysis of methodological approaches for measuring climate adaptation effectiveness, informed by the systematic map.
          </p>
        </div>
      </section>

      {/* Coming Soon */}
      <section className="bg-sand px-6 py-24 flex-grow">
        <div className="max-w-3xl mx-auto text-center">
          <div className="rounded-xl bg-white border border-gray-200 p-12 shadow-sm">
            <h2 className="text-2xl font-logo font-bold text-green mb-4">Coming May–July 2026</h2>
            <p className="font-tagline text-charcoal leading-relaxed mb-6">
              The systematic review and meta-analysis protocol (D6/D7) will be published following completion of the systematic map (D5, target May 2026). The full systematic review manuscript (D8/D9) will follow in June–July 2026.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left mt-8">
              {[
                { id: 'D6', title: 'Draft SR/MA Protocol', due: 'May 2026' },
                { id: 'D7', title: 'Final SR/MA Protocol', due: 'May 2026' },
                { id: 'D8', title: 'Draft SR/MA Manuscript', due: 'Jun 2026' },
                { id: 'D9', title: 'Final SR/MA Manuscript', due: 'Jul 2026' },
              ].map((item) => (
                <div key={item.id} className="rounded-lg border border-gray-200 p-4 flex gap-3 items-start">
                  <span className="font-logo font-bold text-green text-lg">{item.id}</span>
                  <div>
                    <p className="font-tagline font-semibold text-charcoal text-sm">{item.title}</p>
                    <p className="font-tagline text-xs text-gray-400">{item.due}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
