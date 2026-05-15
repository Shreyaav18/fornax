import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Navbar from '@components/Navbar/Navbar'
import PageTransition from '@components/PageTransition/PageTransition'
import Home from '@pages/Home'
import Generate from '@pages/Generate'
import Runs from '@pages/Runs'

function AnimatedRoutes() {
  const location = useLocation()

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route
          path="/"
          element={
            <PageTransition>
              <Home />
            </PageTransition>
          }
        />
        <Route
          path="/generate"
          element={
            <PageTransition>
              <Generate />
            </PageTransition>
          }
        />
        <Route
          path="/runs"
          element={
            <PageTransition>
              <Runs />
            </PageTransition>
          }
        />
        <Route
          path="*"
          element={
            <PageTransition>
              <Home />
            </PageTransition>
          }
        />
      </Routes>
    </AnimatePresence>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <AnimatedRoutes />
    </BrowserRouter>
  )
}