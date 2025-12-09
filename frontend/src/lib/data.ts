// It's good practice to define a type for your data structure
export type TeamMember = {
  name: string;
  role: string;
  image: string;
  slug: string;
  email: string;
  linkedin: string;
  cv: string;
  shortBio: string;
  bio: string;
};

export const teamMembers: TeamMember[] = [
    {
        name: "Jennifer Cissé, PhD",
        role: "Partner",
        image: "/team/jenn.jpg",
        slug: "jenn",
        email: "jenn@bristlep.com",
        linkedin: "https://www.linkedin.com/in/jenncisse/",
        cv: "/cv/Jennifer D Cisse - Resume - Bristlepine.pdf",
        shortBio: "Economist and expert in climate resilience, disaster risk finance, and adaptive development.",
        bio: "Jennifer Cissé is a development economist and specialist in climate resilience, disaster risk finance, and adaptive social protection. With over 15 years of global experience, she has led policy design, program implementation, and evaluation efforts with organizations such as USAID, the United Nations Development Programme (UNDP), the Green Climate Fund (GCF), and the World Bank. Jennifer’s work integrates climate risk into development investments, social protection systems, and disaster preparedness frameworks. She has contributed to national climate finance strategies, adaptive safety net programs, and resilience metrics for vulnerable populations in Africa, Asia, and Latin America. Jennifer holds a Ph.D. in Applied Economics and Management from Cornell University, a Master’s in International Relations from Johns Hopkins SAIS, and a B.A. in Mathematics from Smith College.",
      },
      {
        name: "Caroline Staub, PhD",
        role: "Partner",
        image: "/team/caroline.jpg",
        slug: "caroline",
        email: "caroline@bristlep.com",
        linkedin: "https://www.linkedin.com/in/cgstaub/",
        cv: "/cv/Caroline Staub - Resume - Bristlepine.pdf",
        shortBio: "Geographer and climate adaptation specialist focusing on climate services and resilience.",
        bio: "Caroline Staub is a climate adaptation expert and geographer with a background in agroclimatology and participatory research. Her work focuses on strengthening the use of climate information for decision-making in fragile and climate-vulnerable settings. Caroline has led the design and implementation of large-scale climate services programs funded by USAID, the World Bank, and NASA, and has supported regional and national governments in the development of climate strategies. She brings deep experience in project leadership, research coordination, and capacity building, with a technical foundation in applied climate science. Caroline earned her Ph.D. and M.S. in Geography from the University of Florida and holds a B.Sc. in Environmental Biology from Curtin University in Australia.",
      },
      {
        name: "Zarrar Khan, PhD",
        role: "Partner",
        image: "/team/zarrar.jpg",
        slug: "zarrar",
        email: "zarrar@bristlep.com",
        linkedin: "https://www.linkedin.com/in/khanzarrar/",
        cv: "/cv/Zarrar Khan - Resume - Bristlepine.pdf",
        shortBio: "Systems modeler and climate policy analyst for energy, water, and land sustainability.",
        bio: "Zarrar Khan is a systems modeler and policy advisor focused on the intersection of climate, energy, and sustainable development. He brings technical expertise in integrated assessment modeling, energy system transitions, and climate mitigation strategies, with experience leading projects for USAID, the U.S. Department of Energy, the Inter-American Development Bank (IDB), and the Green Climate Fund. Zarrar has supported governments across Latin America, South Asia, and sub-Saharan Africa in designing data-driven policies for low-carbon development and resilience planning. His work emphasizes cross-sectoral modeling and the development of decision-support tools for national adaptation and energy strategies. Zarrar holds a Ph.D. in Sustainable Energy Technologies and Strategies from KTH Royal Institute of Technology, Comillas University, and TU Delft; a Master of Engineering from Cornell University; and a B.A. in Environmental Studies from Dartmouth College.",
      },
];