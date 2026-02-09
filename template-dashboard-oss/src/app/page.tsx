import Cta from "@/components/landing/Cta"
import Features from "@/components/landing/Features"
import Footer from "@/components/landing/Footer"
import { GlobalDatabase } from "@/components/landing/GlobalDatabase"
import Hero from "@/components/landing/Hero"
import LogoCloud from "@/components/landing/LogoCloud"
import { Navigation } from "@/components/landing/Navbar"

export default function Home() {
  return (
    <>
      <Navigation />
      <main className="flex flex-col overflow-hidden">
        <Hero />
        <LogoCloud />
        <GlobalDatabase />
        <Features />
        <Cta />
      </main>
      <Footer />
    </>
  )
}
