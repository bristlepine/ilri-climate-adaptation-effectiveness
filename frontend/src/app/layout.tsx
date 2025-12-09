import type { Metadata } from "next";
import { EB_Garamond, Lato } from "next/font/google";
import "./globals.css";

const ebGaramond = EB_Garamond({
  variable: "--font-logo",
  subsets: ["latin"],
  weight: ["400", "700"],
});

const lato = Lato({
  variable: "--font-tagline",
  subsets: ["latin"],
  weight: ["400", "700"],
});

export const metadata: Metadata = {
  title: "ILRI – Measuring What Matters",
  description:
    "Tracking the effectiveness of climate adaptation for smallholder producers through evidence synthesis, systematic reviews, and systematic mapping.",

  icons: {
    icon: "/logo_bg_black.ico",
  },

  openGraph: {
    title: "ILRI – Measuring What Matters",
    description:
      "Evidence-synthesis project assessing climate adaptation effectiveness in smallholder agricultural systems.",
    url: "https://ilri-climate-adaptation-effectiveness.org",
    images: [
      {
        url: "/og.png",
        width: 1200,
        height: 630,
        alt: "ILRI Climate Adaptation Effectiveness Project",
      },
    ],
  },

  twitter: {
    card: "summary_large_image",
    title: "ILRI – Measuring What Matters",
    description:
      "Tracking the effectiveness of climate adaptation for smallholder producers.",
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${ebGaramond.variable} ${lato.variable}`}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
