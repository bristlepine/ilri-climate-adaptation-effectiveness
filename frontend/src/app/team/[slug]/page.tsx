'use client';

import { useParams } from 'next/navigation';
import Link from 'next/link';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { FaLinkedin } from 'react-icons/fa';
import { HiOutlineMail } from 'react-icons/hi';
import { IoDocumentTextOutline } from 'react-icons/io5';

// 1. Import the single source of truth for your data
import { teamMembers } from '@/lib/data';

export default function TeamMemberPage() {
    // 2. Get the slug from the URL.
    // If you go to /team/jenn, `slug` will be "jenn".
    const { slug } = useParams();

    // 3. Find the specific team member in the array whose slug matches the one from the URL.
    const member = teamMembers.find(m => m.slug === slug);

    // 4. If no member is found (e.g., bad URL), show a "Not Found" message.
    if (!member) {
        return (
            <main className="page-wrapper">
                <Navbar />
                <section className="px-6 py-48 text-center">
                    <h1 className="text-2xl font-logo text-green mb-4">Team Member Not Found</h1>
                    <p className="max-w-2xl mx-auto text-lg mb-8">
                        The profile you are looking for does not exist.
                    </p>
                    <Link href="/team" className="text-green hover:underline">
                        &larr; Back to Team Page
                    </Link>
                </section>
                <Footer />
            </main>
        );
    }

    // 5. If a member IS found, display their information using this template.
    return (
        <main className="page-wrapper">
            <Navbar />
            <section className="px-6 py-24">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center md:text-left md:flex md:items-center md:gap-12">
                        <img
                            src={member.image}
                            alt={member.name}
                            className="w-48 h-48 rounded-full mb-8 md:mb-0 mx-auto md:mx-0 border-2 border-green shadow-lg flex-shrink-0"
                        />
                        <div className="flex-grow">
                            <h1 className="text-4xl font-logo text-green mb-2">{member.name}</h1>
                            <p className="text-xl font-tagline text-charcoal mb-6">{member.role}</p>
                            <div className="flex justify-center md:justify-start gap-6 text-green text-3xl">
                                <a href={member.linkedin} target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                                    <FaLinkedin className="hover:opacity-70" />
                                </a>
                                <a href={`mailto:${member.email}`} aria-label="Email">
                                    <HiOutlineMail className="hover:opacity-70" />
                                </a>
                                <a href={member.cv} target="_blank" rel="noopener noreferrer" aria-label="CV">
                                    <IoDocumentTextOutline className="hover:opacity-70" />
                                </a>
                            </div>
                        </div>
                    </div>
                    <div className="mt-12 pt-8 border-t border-gray-200">
                        <p className="text-charcoal text-base leading-relaxed whitespace-pre-line">
                            {member.bio}
                        </p>
                    </div>
                    <div className="mt-16 text-center">
                        <Link href="/team" className="text-green hover:underline font-semibold">
                            &larr; Back to Team Page
                        </Link>
                    </div>
                </div>
            </section>
            <Footer />
        </main>
    );
}