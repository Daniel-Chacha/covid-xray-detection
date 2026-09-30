import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Disclaimer from "@/components/Disclaimer";
import SiteNav from "@/components/SiteNav";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "How much of this COVID classifier is real?",
  description:
    "Two DenseNet121 models run in your browser on the same chest X-ray — one on the full image, one with the lungs erased. The predictions barely differ.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-neutral-950 font-sans text-neutral-100">
        <Disclaimer />
        <SiteNav />
        {children}
      </body>
    </html>
  );
}
