import type { Metadata } from "next";
import { RootProvider } from 'fumadocs-ui/provider/next';
import "./globals.css";

export const metadata: Metadata = {
  title: "Ultrabull Pipeline Docs",
  description: "Technical and statistical documentation for the ultrabull stock analysis pipeline",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="flex flex-col min-h-screen antialiased">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
