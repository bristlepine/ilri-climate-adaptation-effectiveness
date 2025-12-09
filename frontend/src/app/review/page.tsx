import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export default function ServicesPage() {
  return (
    <main className="page-wrapper min-h-screen flex flex-col">
      <Navbar />

      {/* Page content */}
      <section className="px-6 py-24 text-center flex-grow">
        <h1 className="text-4xl font-logo text-green mb-6">Our Services</h1>
        <p className="max-w-2xl mx-auto text-lg">
          Discover what Bristlepine can offer to help your organization grow and thrive.
        </p>
      </section>

      <Footer />
    </main>
  );
}
