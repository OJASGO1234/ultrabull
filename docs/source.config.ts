import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import { remarkMdxMermaid, remarkGfm } from 'fumadocs-core/mdx-plugins';

export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    mdxOptions: {
      remarkPlugins: [remarkGfm, remarkMdxMermaid],
    },
  },
});

export default defineConfig();
