import { useRef } from 'react'
import { motion, useScroll, useTransform } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { useMouseParallax } from '@hooks/useParallax'
import useMagneticHover from '@hooks/useMagneticHover'
import styles from './HeroSection.module.css'

const WORD_VARIANTS = {
  initial: { opacity: 0, y: 40, filter: 'blur(12px)' },
  animate: (i) => ({
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: {
      duration: 0.7,
      delay: i * 0.08,
      ease: [0.25, 0.46, 0.45, 0.94]
    }
  })
}

function AnimatedHeading({ text }) {
  const words = text.split(' ')
  return (
    <h1 className={styles.heading}>
      {words.map((word, i) => (
        <motion.span
          key={i}
          className={styles.word}
          custom={i}
          variants={WORD_VARIANTS}
          initial="initial"
          animate="animate"
        >
          {word}
        </motion.span>
      ))}
    </h1>
  )
}

export default function HeroSection() {
  const containerRef = useRef(null)
  const navigate = useNavigate()
  const ctaRef = useMagneticHover({ strength: 0.25, smoothing: 0.1 })
  const mousePos = useMouseParallax({ strength: 0.015, smoothing: 0.08 })

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end start']
  })

  const heroY = useTransform(scrollYProgress, [0, 1], [0, 180])
  const heroOpacity = useTransform(scrollYProgress, [0, 0.6], [1, 0])
  const auroraScale = useTransform(scrollYProgress, [0, 1], [1, 1.3])
  const auroraY = useTransform(scrollYProgress, [0, 1], [0, 120])

  return (
    <section ref={containerRef} className={styles.section}>
      <motion.div
        className={styles.aurora}
        style={{ scale: auroraScale, y: auroraY }}
      >
        <motion.div
          className={styles.auroraBlob1}
          animate={{
            x: [0, 40, -20, 0],
            y: [0, -30, 20, 0],
            scale: [1, 1.1, 0.95, 1]
          }}
          transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
          style={{
            x: mousePos.x * 1.5,
            y: mousePos.y * 1.5
          }}
        />
        <motion.div
          className={styles.auroraBlob2}
          animate={{
            x: [0, -50, 30, 0],
            y: [0, 40, -20, 0],
            scale: [1, 0.9, 1.15, 1]
          }}
          transition={{ duration: 15, repeat: Infinity, ease: 'easeInOut', delay: 2 }}
          style={{
            x: mousePos.x * -1,
            y: mousePos.y * -1
          }}
        />
        <motion.div
          className={styles.auroraBlob3}
          animate={{
            x: [0, 30, -40, 0],
            y: [0, -20, 40, 0],
            scale: [1, 1.2, 0.9, 1]
          }}
          transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut', delay: 4 }}
          style={{
            x: mousePos.x * 0.8,
            y: mousePos.y * -0.8
          }}
        />
      </motion.div>

      <div className={styles.noise} />

      <motion.div
        className={styles.content}
        style={{ y: heroY, opacity: heroOpacity }}
      >
        <motion.div
          className={styles.badge}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        >
          <motion.span
            className={styles.badgeDot}
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <span className={styles.badgeText}>Transformer built from scratch</span>
        </motion.div>

        <AnimatedHeading text="Language that thinks from the ground up" />

        <motion.p
          className={styles.subheading}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
        >
          Fornax is a GPT-style language model built entirely from first principles.
          No black boxes. Pure architecture, pure understanding.
        </motion.p>

        <motion.div
          className={styles.actions}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
        >
          <motion.button
            ref={ctaRef}
            className={styles.ctaPrimary}
            onClick={() => navigate('/generate')}
            whileTap={{ scale: 0.97 }}
          >
            <span className={styles.ctaText}>Start Generating</span>
            <motion.span
              className={styles.ctaGlow}
              animate={{ opacity: [0.4, 0.8, 0.4] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
            />
          </motion.button>

          <motion.button
            className={styles.ctaSecondary}
            onClick={() => navigate('/runs')}
            whileHover={{ borderColor: 'rgba(139, 92, 246, 0.4)' }}
            whileTap={{ scale: 0.97 }}
          >
            View Runs
          </motion.button>
        </motion.div>

        <motion.div
          className={styles.stats}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 1.1 }}
        >
          {[
            { value: 'RoPE', label: 'Positional Encoding' },
            { value: 'SwiGLU', label: 'Activation' },
            { value: 'RMSNorm', label: 'Normalization' },
            { value: 'BPE', label: 'Tokenizer' }
          ].map((stat, i) => (
            <motion.div
              key={stat.label}
              className={styles.stat}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 1.1 + i * 0.07 }}
            >
              <span className={`${styles.statValue} gradient-text`}>{stat.value}</span>
              <span className={styles.statLabel}>{stat.label}</span>
            </motion.div>
          ))}
        </motion.div>
      </motion.div>

      <motion.div
        className={styles.scrollIndicator}
        animate={{ y: [0, 8, 0], opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
      >
        <div className={styles.scrollLine} />
      </motion.div>
    </section>
  )
}