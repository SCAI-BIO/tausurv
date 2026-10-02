import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const base = "/tausurv";

// Astro's `base` does not apply to links written in Markdown, so prefix
// root-relative link and image URLs here.
function remarkBasePath() {
  const walk = (node) => {
    const linked = node.type === "link" || node.type === "image" || node.type === "definition";
    if (linked && node.url.startsWith("/") && !node.url.startsWith("//")) {
      node.url = base + node.url;
    }
    node.children?.forEach(walk);
  };
  return walk;
}

export default defineConfig({
  site: "https://scai-bio.github.io",
  base,
  markdown: {
    remarkPlugins: [remarkMath, remarkBasePath],
    rehypePlugins: [[rehypeKatex, {}]],
  },
  integrations: [
    starlight({
      title: "tausurv",
      description: "Modern survival and causal-survival analysis for Python.",
      customCss: [
        "katex/dist/katex.min.css",
        "./src/styles/custom.css",
      ],
      sidebar: [
        { label: "Start", items: [
          { label: "Overview", slug: "index" },
          { label: "Installation", slug: "installation" },
          { label: "Quickstart", slug: "quickstart" },
        ]},
        { label: "Tutorials", items: [{ autogenerate: { directory: "tutorials" } }] },
        { label: "How-to", items: [{ autogenerate: { directory: "how-to" } }] },
        { label: "Concepts", items: [{ autogenerate: { directory: "concepts" } }] },
        { label: "Plotting", items: [{ autogenerate: { directory: "plotting" } }] },
        { label: "API reference", items: [{ autogenerate: { directory: "api" } }] },
      ],
    }),
  ],
});
