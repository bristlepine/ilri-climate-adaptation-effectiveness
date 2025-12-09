"use client";

import { useState } from "react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { ArrowRight, X } from "lucide-react";
import projects from "@/lib/projects.json"; // ✅ pulling from JSON

type Project = {
  title: string;
  institution: string;
  subtitle: string;
  link: string;
  image: string;
  status?: string; // now optional
  imageLink?: string; // allow this field
  startDate?: string;
  endDate?: string | null;
};

function formatDate(dateString?: string | null): string {
  if (!dateString) return "Present";
  const date = new Date(dateString);
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short", // e.g., Jan, Feb, Mar
  });
}

export default function ProjectsPage() {
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);

  return (
    <main className="page-wrapper relative">
      <Navbar />
      <section
        id="projects"
        className="relative min-h-[calc(80vh)] px-6 pt-32 pb-20 text-sand overflow-hidden"
      >
        {/* Background Video */}
        <video
          autoPlay
          muted
          loop
          playsInline
          className="absolute inset-0 w-full h-full object-cover z-[-1]"
        >
          <source
            src="https://bristlepine.s3.us-east-2.amazonaws.com/hero1.webm"
            type="video/webm"
          />
        </video>

        {/* Dark overlay */}
        <div className="absolute inset-0 bg-black/60 z-0" />

        <div className="relative z-10 mt-10 mb-16 max-w-6xl mx-auto text-center">
          <h2 className="text-4xl font-logo font-bold mb-6">Our Experience</h2>
          <p className="text-lg font-tagline font-medium mb-12">
            Relevant projects from Bristlepine and our consultants' past work.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-10 px-4">
            {projects.map((project, index) => (
              <div
                key={index}
                onClick={() => setSelectedProject(project)}
                className="cursor-pointer group rounded-xl overflow-hidden bg-sand text-charcoal shadow-[0_2px_12px_rgba(202,194,181,0.25)] hover:shadow-[0_4px_16px_rgba(202,194,181,0.4)] transition-all duration-300 transform hover:scale-[1.015] flex flex-col"
              >
                <div className="relative h-48 overflow-hidden">
                  <img
                    src={project.image}
                    alt={project.title}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                  <div className="absolute inset-0 bg-black/40 group-hover:bg-black/50 transition-all duration-300 flex items-center justify-center">
                    <h3 className="text-lg font-bold text-sand text-center px-4">
                      {project.title}
                    </h3>
                  </div>
                </div>
                <div className="flex flex-col justify-between flex-1 p-4 text-left">
                  {/* Institution */}
                  <p className="text-xs italic text-gray-500 mb-1">
                    {project.institution}
                  </p>
                  {/* Dates */}
                  {(project.startDate || project.endDate) && (
                    <p className="text-xs text-gray-400 mb-2">
                      {formatDate(project.startDate)} –{" "}
                      {project.endDate ? formatDate(project.endDate) : "Present"}
                    </p>
                  )}
                  <p className="text-sm font-tagline mb-3 line-clamp-3">
                    {project.subtitle}
                  </p>
                  <div className="flex items-center gap-2 text-green group-hover:text-sage transition-colors duration-300">
                    <ArrowRight className="w-5 h-5" />
                    <span className="text-sm font-semibold">Learn More</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      <Footer />

      {/* Modal */}
      {selectedProject && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-white text-charcoal rounded-xl max-w-lg w-full shadow-xl relative p-6">
            {/* Close button */}
            <button
              onClick={() => setSelectedProject(null)}
              className="absolute top-3 right-3 text-gray-500 hover:text-black"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Project content */}
            <h3 className="text-xl font-bold mb-2">{selectedProject.title}</h3>
            <p className="text-sm italic text-gray-600 mb-2">
              {selectedProject.institution}
            </p>

            {/* Dates */}
            {(selectedProject.startDate || selectedProject.endDate) && (
              <p className="text-xs text-gray-500 mb-4">
                {formatDate(selectedProject.startDate)} –{" "}
                {selectedProject.endDate
                  ? formatDate(selectedProject.endDate)
                  : "Present"}
              </p>
            )}

            <p className="text-base mb-6">{selectedProject.subtitle}</p>

            {selectedProject.link && selectedProject.link !== "#" && (
              <a
                href={selectedProject.link}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-block px-4 py-2 bg-green text-white rounded-lg hover:bg-sage transition"
              >
                Learn More
              </a>
            )}
          </div>
        </div>
      )}
    </main>
  );
}
