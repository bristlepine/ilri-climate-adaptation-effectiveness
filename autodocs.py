#!/usr/bin/env python3
"""
autodocs.py — updates README structure AND generates frontend About page.

Rules:
- Show all top-level folders/files.
- For all folders: show only ONE level of files.
- EXCEPTION: deliverables/ goes one level deeper.
- Also converts README.md → safe HTML and writes to:
    frontend/src/app/about/page.tsx
"""

from pathlib import Path
import re
import html

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
FRONTEND_PAGE = ROOT / "frontend/src/app/about/page.tsx"

SECTION_START = "<!-- AUTO-STRUCTURE:START -->"
SECTION_END   = "<!-- AUTO-STRUCTURE:END -->"


# -----------------------------------------------------------------------------
# REPO STRUCTURE
# -----------------------------------------------------------------------------

def indent(level: int) -> str:
    return "│   " * level

def generate_structure(path: Path, level: int = 0) -> list:
    lines = []
    for item in sorted(path.iterdir()):
        if item.name.startswith(".") or item.name == "README.md":
            continue

        if level == 0:
            if item.is_dir():
                lines.append(f"├── {item.name}/")

                # Normal folders — one level of files only
                if item.name != "deliverables":
                    for f in sorted(item.iterdir()):
                        if f.is_file() and not f.name.startswith("."):
                            lines.append(f"{indent(1)}├── {f.name}")
                    continue

                # deliverables: one level deeper
                for sub in sorted(item.iterdir()):
                    if sub.is_dir():
                        lines.append(f"{indent(1)}├── {sub.name}/")
                        for f in sorted(sub.iterdir()):
                            if f.is_file() and not f.name.startswith("."):
                                lines.append(f"{indent(2)}├── {f.name}")
            else:
                lines.append(f"├── {item.name}")
    return lines

def get_repo_structure(root: Path) -> str:
    lines = ["```bash", "repo-root/"]
    lines.extend(generate_structure(root))
    lines.append("```")
    return "\n".join(lines)

def update_readme(structure_block: str):
    text = README.read_text()

    if SECTION_START not in text or SECTION_END not in text:
        raise ValueError("README.md missing AUTO-STRUCTURE comment markers.")

    new_text = []
    inside = False
    for line in text.splitlines():
        if SECTION_START in line:
            inside = True
            new_text.append(line)
            new_text.append(structure_block)
            continue
        if SECTION_END in line:
            inside = False
            new_text.append(line)
            continue
        if not inside:
            new_text.append(line)

    README.write_text("\n".join(new_text) + "\n")


# -----------------------------------------------------------------------------
# README → SAFE HTML (minimal markdown)
# -----------------------------------------------------------------------------

def md_to_html(md: str) -> str:
    # Strip HTML comments (incl. AUTO-STRUCTURE markers)
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    md = md.replace("\r\n", "\n")

    # Handle fenced code blocks first and keep placeholders
    code_blocks = []
    def _code_repl(m):
        code = m.group(1)
        code_escaped = html.escape(code)
        idx = len(code_blocks)
        code_blocks.append(
            f'<pre class="p-4 bg-gray-900 text-gray-100 rounded-md my-4 overflow-auto font-mono text-sm"><code>{code_escaped}</code></pre>'
        )
        return f"[[[CODE_BLOCK_{idx}]]]"

    md = re.sub(r"```(.*?)```", _code_repl, md, flags=re.DOTALL)

    # Images: ![alt](url)
    md = re.sub(
        r"!\[(.*?)\]\((.*?)\)",
        lambda m: f'<img alt="{html.escape(m.group(1))}" src="{m.group(2)}" class="my-4 rounded" />',
        md,
    )

    # Links: [text](url)
    md = re.sub(
        r"\[(.*?)\]\((.*?)\)",
        lambda m: f'<a href="{m.group(2)}" class="text-green underline">{html.escape(m.group(1))}</a>',
        md,
    )

    # Headings
    md = re.sub(r"^### (.*)$", lambda m: f'<h3 class="text-xl font-semibold mt-6 mb-2">{html.escape(m.group(1))}</h3>', md, flags=re.MULTILINE)
    md = re.sub(r"^## (.*)$",  lambda m: f'<h2 class="text-2xl font-bold mt-10 mb-4">{html.escape(m.group(1))}</h2>', md, flags=re.MULTILINE)
    md = re.sub(r"^# (.*)$",   lambda m: f'<h1 class="text-3xl font-bold mt-12 mb-6">{html.escape(m.group(1))}</h1>', md, flags=re.MULTILINE)

    # Bold and inline code
    md = re.sub(r"\*\*(.*?)\*\*", lambda m: f"<strong>{html.escape(m.group(1))}</strong>", md)
    md = re.sub(r"`([^`\n]+)`",   lambda m: f'<code class="px-1 py-0.5 bg-gray-200 rounded text-sm font-mono">{html.escape(m.group(1))}</code>', md)

    # List items (- or *)
    md = re.sub(r"^(?:-|\*) (.*)$", lambda m: f"<li>{html.escape(m.group(1))}</li>", md, flags=re.MULTILINE)

    # Wrap consecutive <li> blocks into a <ul>
    def wrap_ul(match):
        block = match.group(0)
        return f'<ul class="my-4 list-disc ml-6">\n{block}\n</ul>\n'
    md = re.sub(r"(?:<li>.*?</li>\n?)+", wrap_ul, md, flags=re.DOTALL)

    # Turn loose lines into paragraphs (escape them)
    out_lines = []
    for line in md.split("\n"):
        s = line.strip()
        if not s:
            out_lines.append("")
            continue
        if s.startswith("<"):     # already an HTML element we created
            out_lines.append(line)
        else:
            out_lines.append(f'<p class="my-3 leading-relaxed">{html.escape(line)}</p>')
    html_out = "\n".join(out_lines)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        html_out = html_out.replace(f"[[[CODE_BLOCK_{i}]]]", block)

    return html_out


# -----------------------------------------------------------------------------
# WRITE ABOUT PAGE (render HTML safely)
# -----------------------------------------------------------------------------

def write_about_page(readme_html: str):
    # Escape backticks and backslashes so we can put the HTML inside a JS template literal
    html_for_js = readme_html.replace("\\", "\\\\").replace("`", "\\`")

    FRONTEND_PAGE.parent.mkdir(parents=True, exist_ok=True)

    template = (
        "'use client';\n\n"
        "import Navbar from \"@/components/Navbar\";\n"
        "import Footer from \"@/components/Footer\";\n"
        "import { motion } from \"framer-motion\";\n\n"
        "export default function AboutPage() {\n"
        "  const README_HTML = `"
        + html_for_js +
        "`;\n"
        "  return (\n"
        "    <main className=\"page-wrapper\">\n"
        "      <Navbar />\n\n"
        "      {/* Hero Section */}\n"
        "      <section className=\"relative h-[60vh] w-full text-white overflow-hidden\">\n"
        "        <video autoPlay muted loop playsInline className=\"absolute inset-0 w-full h-full object-cover z-[-1]\">\n"
        "          <source src=\"https://bristlepine.s3.us-east-2.amazonaws.com/2994205-uhd_3840_2160_30fps_clip.mp4\" type=\"video/webm\" />\n"
        "        </video>\n"
        "        <div className=\"absolute inset-0 bg-black/60 z-0\" />\n"
        "        <div className=\"relative z-10 flex flex-col justify-center items-center text-center px-6 pt-20 h-full\">\n"
        "          <h1 className=\"text-5xl font-logo text-sand mb-4\">About</h1>\n"
        "          <p className=\"text-lg font-tagline text-white/90 max-w-2xl mx-auto\">Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector.</p>\n"
        "        </div>\n"
        "      </section>\n\n"
        "      {/* README Content */}\n"
        "      <section className=\"px-6 py-16 bg-white text-charcoal\">\n"
        "        <div className=\"max-w-4xl mx-auto prose prose-lg\">\n"
        "          <div dangerouslySetInnerHTML={{ __html: README_HTML }} />\n"
        "        </div>\n"
        "      </section>\n\n"
        "      <Footer />\n"
        "    </main>\n"
        "  );\n"
        "}\n"
    )

    FRONTEND_PAGE.write_text(template)
    print(f"Updated frontend About page → {FRONTEND_PAGE}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Updating README structure…")
    structure = get_repo_structure(ROOT)
    update_readme(structure)

    print("Converting README → safe HTML…")
    readme_html = md_to_html(README.read_text())

    print("Writing front-end About page…")
    write_about_page(readme_html)

    print("✔ autodocs.py completed successfully.")
