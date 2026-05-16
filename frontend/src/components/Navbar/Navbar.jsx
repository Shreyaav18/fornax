import { useState, useEffect } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { motion, useScroll, useTransform } from 'framer-motion'
import useMagneticHover from '@hooks/useMagneticHover'
import styles from './Navbar.module.css'

const LINKS = [
  { path: '/', label: 'Home' },
  { path: '/generate', label: 'Generate' },
  { path: '/runs', label: 'Runs' }
]

function MagneticLink({ path, label }) {
  const ref = useMagneticHover({ strength: 0.2, smoothing: 0.12 })
  const location = useLocation()
  const isActive = location.pathname === path

  return (
    <NavLink to={path} className={styles.linkWrapper}>
      <motion.span
        ref={ref}
        className={styles.link}
        data-active={isActive}
        whileTap={{ scale: 0.95 }}
      >
        {label}
        {isActive && (
          <motion.span
            className={styles.activeDot}
            layoutId="activeDot"
            transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
          />
        )}
      </motion.span>
    </NavLink>
  )
}

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const { scrollY } = useScroll()

  const navBlur = useTransform(scrollY, [0, 80], [0, 16])
  const navOpacity = useTransform(scrollY, [0, 80], [0, 0.8])

  useEffect(() => {
    return scrollY.on('change', (v) => setScrolled(v > 40))
  }, [scrollY])

  return (
    <motion.nav
      className={styles.nav}
      data-scrolled={scrolled}
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
    >
      <motion.div
        className={styles.backdrop}
        style={{ backdropFilter: useTransform(navBlur, (v) => `blur(${v}px)`) }}
      >
        <motion.div
          className={styles.backdropBg}
          style={{ opacity: navOpacity }}
        />
      </motion.div>

      <div className={styles.inner}>
        <NavLink to="/" className={styles.logoWrapper}>
          <motion.div
            className={styles.logo}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
          >
            <span className={`${styles.logoText} gradient-text`}>FORNAX</span>
            <span className={styles.logoSub}>LM</span>
          </motion.div>
        </NavLink>

        <div className={styles.links}>
          {LINKS.map((link) => (
            <MagneticLink key={link.path} {...link} />
          ))}
        </div>

        <motion.div
          className={styles.pill}
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        >
          <span className={styles.pillDot} />
          <span className={styles.pillLabel}>v1.0</span>
        </motion.div>
      </div>
    </motion.nav>
  )
}