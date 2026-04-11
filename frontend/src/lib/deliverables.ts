export type DeliverableStatus =
  | 'submitted'
  | 'in-progress'
  | 'overdue'
  | 'pending'
  | 'not-started';

export type Deliverable = {
  id: string;
  title: string;
  subtitle: string;
  type: string;
  description: string;
  due: string;
  status: DeliverableStatus;
  zenodo?: string;
  github?: string;
  cgspace?: string;
  note?: string;
};

export const deliverables: Deliverable[] = [
  {
    id: 'D1',
    title: 'Inception Report',
    subtitle: '(Interim)',
    type: 'Report',
    description:
      'Final research questions and PCCM framework, overall approach to the review, proposed search strings, volume of literature surfaced, and a detailed Gantt chart showing all review steps.',
    due: 'Nov 2025',
    status: 'submitted',
    zenodo: 'https://doi.org/10.5281/zenodo.17861055',
    github: 'https://github.com/bristlepine/ilri-climate-adaptation-effectiveness',
  },
  {
    id: 'D2',
    title: 'First Draft Scoping Protocol',
    subtitle: '(Interim)',
    type: 'Report',
    description:
      'First draft of the protocol for the scoping review and systematic map, including draft PCCM eligibility criteria and methodology appendix.',
    due: 'Dec 2025',
    status: 'submitted',
    zenodo: 'https://zenodo.org/records/18369383',
    github: 'https://github.com/bristlepine/ilri-climate-adaptation-effectiveness',
  },
  {
    id: 'D3',
    title: 'Final Scoping Protocol',
    subtitle: '(Final)',
    type: 'Report',
    description:
      'Final protocol for the scoping review and systematic map, published on Zenodo with a citable DOI. Includes finalised eligibility criteria, search strings, and methodology appendix.',
    due: 'Jan 2026',
    status: 'submitted',
    zenodo: undefined,
    github: 'https://github.com/bristlepine/ilri-climate-adaptation-effectiveness',
    cgspace: undefined,
    note: 'Zenodo DOI to be added.',
  },
  {
    id: 'D4',
    title: 'Draft Systematic Map',
    subtitle: '(Interim)',
    type: 'Report + Searchable Database',
    description:
      'First draft of the scoping review and systematic map based on Scopus corpus (~17,000 records). Includes a preliminary searchable database of included evidence and evidence gap map. Labelled preliminary pending multi-database integration.',
    due: 'Apr 2026',
    status: 'in-progress',
    zenodo: undefined,
    github: undefined,
    note: 'Scopus-based only; multi-database integration (WoS, CAB, AGRIS, ASP) pending.',
  },
  {
    id: 'D5',
    title: 'Final Systematic Map',
    subtitle: '(Final)',
    type: 'Report + Searchable Database',
    description:
      'Final scoping review and systematic map incorporating all databases, grey literature, and full-text screening. Includes the complete searchable extraction database and evidence gap map, published on Zenodo and CGSpace.',
    due: 'May 2026',
    status: 'not-started',
    zenodo: undefined,
    github: undefined,
    cgspace: undefined,
  },
  {
    id: 'D6',
    title: 'Draft SR/Meta-Analysis Protocol',
    subtitle: '(Interim)',
    type: 'Report',
    description:
      'First draft of the protocol for the systematic review and meta-analysis, informed by the systematic map findings. Includes scope, research questions, inclusion criteria, and analysis plan.',
    due: 'May 2026',
    status: 'not-started',
    zenodo: undefined,
    github: undefined,
  },
  {
    id: 'D7',
    title: 'Final SR/Meta-Analysis Protocol',
    subtitle: '(Final)',
    type: 'Report',
    description:
      'Final protocol for the systematic review and meta-analysis, published on Zenodo with a citable DOI and submitted to CGSpace.',
    due: 'May 2026',
    status: 'not-started',
    zenodo: undefined,
    github: undefined,
    cgspace: undefined,
  },
  {
    id: 'D8',
    title: 'Draft Systematic Review / Meta-Analysis',
    subtitle: '(Interim)',
    type: 'Journal Paper',
    description:
      'First draft of the systematic review and meta-analysis ready for submission to a peer-reviewed journal. Includes effect size extraction, meta-analysis, and evidence synthesis.',
    due: 'Jun 2026',
    status: 'not-started',
    zenodo: undefined,
    github: undefined,
  },
  {
    id: 'D9',
    title: 'Final Systematic Review / Meta-Analysis',
    subtitle: '(Final)',
    type: 'Journal Paper',
    description:
      'Final draft of the systematic review and meta-analysis, incorporating reviewer feedback and ready for journal submission.',
    due: 'Jul 2026',
    status: 'not-started',
    zenodo: undefined,
    github: undefined,
  },
  {
    id: 'D10',
    title: 'Summary Presentation',
    subtitle: '(Final)',
    type: 'PowerPoint',
    description:
      'PowerPoint presentation summarising all outputs — protocols, systematic map, systematic review — and key findings for a lay audience. Formatted for ILRI.',
    due: 'Jul 2026',
    status: 'not-started',
    zenodo: undefined,
    github: undefined,
  },
];
