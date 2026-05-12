import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMath],
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
