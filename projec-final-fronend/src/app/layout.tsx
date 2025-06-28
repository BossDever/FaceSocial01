import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import ConsoleWarningSuppress from "@/components/ConsoleWarningSuppress";
import AntdConfigProvider from "@/components/AntdConfigProvider";
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
  title: "AI Testing Hub - Computer Vision Testing Platform",
  description: "ทดสอบและประเมินประสิทธิภาพของระบบ AI Computer Vision รองรับ Face Detection, Anti-Spoofing, Age-Gender Analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ConsoleWarningSuppress />
        <AntdConfigProvider>
          {children}
        </AntdConfigProvider>
      </body>
    </html>
  );
}
