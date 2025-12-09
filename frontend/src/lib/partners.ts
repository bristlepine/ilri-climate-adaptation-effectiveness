export type Partner = {
  name: string;
  logo: string; // Path to the logo in the /public folder
};

export const partners: Partner[] = [
  { name: "IFAD", logo: "/logos/ifad.png" },
  { name: "World Bank", logo: "/logos/worldbank.png" },
  { name: "AATI", logo: "/logos/aati.png" },
  { name: "unuehs", logo: "/logos/unuehs.svg" },
  { name: "mcii", logo: "/logos/mcii.avif" },
  { name: "reading", logo: "/logos/reading.webp" },
];