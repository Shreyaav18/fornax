import { useRef } from 'react'
import { motion, useScroll, useTransform } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import HeroSection from '@components/HeroSection/HeroSection'
import styles from './Home.module.css'

const FEATURES = [
  {
    tag: 'Architecture',
    title: 'Built from first principles',
    description: 'Every layer written by hand. Attention, FFN, RoPE, RMSNorm — no abstraction left unexplained.'
  },
  {
    tag: 'Tokenizer',
    title: 'Byte Pair Encoding',
    description: 'A BPE tokenizer trained directly on your corpus. Vocabulary shaped by your data, not someone else\'s.'
  },
  {
    tag: 'Training',
    title: 'AdamW with cosine decay',
    description: 'Warmup, gradient clipping, label smoothing, gradient accumulation. Production-grade training loop at any scale.'
  },
  {
    tag: 'Inference',
    title: 'KV Cache + nucleus sampling',
    description: 'Fast autoregressive generation with top-k, top-p, temperature scaling, and repetition penalty built in.'
  }
]

function FeatureCard({ feature, index }) {
  const ref = useRef(null)
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start 0.9', 'start 0.4']
  })

  const opacity = useTransform(scrollYProgress, [0, 1], [0, 1])
  const y = useTransform(scrollYProgress, [0, 1], [48, 0])
  const blur = useTransform(scrollYProgress, [0, 1], [8, 0])

  return (
    <motion.div
      ref={ref}
      className={`${styles.featureCard} gradient-border`}
      style={{ opacity, y, filter: useTransform(blur, (v) => `blur(${v}px)`) }}
    >
      <span className={styles.featureTag}>{feature.tag}</span>
      <h3 className={styles.featureTitle}>{feature.title}</h3>
      <p className={styles.featureDesc}>{feature.description}</p>
      <motion.div
        className={styles.featureGlow}
        animate={{ opacity: [0, 0.6, 0] }}
        transition={{
          duration: 3,
          repeat: Infinity,
          delay: index * 0.4,
          ease: 'easeInOut'
        }}
      />
    </motion.div>
  )
}

function MarqueeText() {
  const ITEMS = [
    'TRANSFORMER', 'FORNAX', 'ATTENTION', 'ROPE', 'SWIGLU',
    'RMSNORM', 'BPE', 'ADAMW', 'KV CACHE', 'NUCLEUS SAMPLING'
  ]

  return (
    <div className={styles.marqueeWrapper}>
      <motion.div
        className={styles.marqueeTrack}
        animate={{ x: ['0%', '-50%'] }}
        transition={{ duration: 24, repeat: Infinity, ease: 'linear' }}
      >
        {[...ITEMS, ...ITEMS].map((item, i) => (
          <span key={i} className={styles.marqueeItem}>
            <span className={`${styles.marqueeText} gradient-text`}>{item}</span>
            <span className={styles.marqueeDot} />
          </span>
        ))}
      </motion.div>
    </div>
  )
}

export default function Home() {
  const navigate = useNavigate()
  const ctaRef = useRef(null)

  const { scrollYProgress } = useScroll()
  const ctaOpacity = useTransform(scrollYProgress, [0.7, 0.85], [0, 1])
  const ctaY = useTransform(scrollYProgress, [0.7, 0.85], [40, 0])

  return (
    <main className={styles.main}>
      <HeroSection />

      <MarqueeText />

      <section className={styles.featuresSection}>
        <motion.div
          className={styles.sectionHeader}
          initial={{ opacity: 0, y: 32 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
        >
          <span className={styles.sectionTag}>What is Fornax</span>
          <h2 className={styles.sectionTitle}>
            Every component,
            <br />
            <span className="gradient-text">understood deeply</span>
          </h2>
        </motion.div>

        <div className={styles.featuresGrid}>
          {FEATURES.map((feature, i) => (
            <FeatureCard key={feature.tag} feature={feature} index={i} />
          ))}
        </div>
      </section>

      <section className={styles.architectureSection}>
        <motion.div
          className={styles.archFlow}
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.8 }}
        >
          {[
            'Raw Text',
            'BPE Tokens',
            'Embeddings',
            'RoPE + Attention',
            'SwiGLU FFN',
            'RMSNorm',
            'Logits'
          ].map((step, i) => (
            <motion.div
              key={step}
              className={styles.archStep}
              initial={{ opacity: 0, x: -24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{
                duration: 0.5,
                delay: i * 0.08,
                ease: [0.25, 0.46, 0.45, 0.94]
              }}
            >
              <div className={styles.archNode}>
                <span className={styles.archNodeText}>{step}</span>
              </div>
              {i < 6 && (
                <motion.div
                  className={styles.archConnector}
                  initial={{ scaleX: 0 }}
                  whileInView={{ scaleX: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.08 + 0.2 }}
                />
              )}
            </motion.div>
          ))}
        </motion.div>
      </section>

      <motion.section
        className={styles.ctaSection}
        style={{ opacity: ctaOpacity, y: ctaY }}
      >
        <div className={styles.ctaGlow} />
        <motion.h2
          className={styles.ctaTitle}
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          Ready to generate?
        </motion.h2>
        <motion.p
          className={styles.ctaSubtitle}
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          Pick a trained run and start prompting Fornax.
        </motion.p>
        <motion.button
          ref={ctaRef}
          className={styles.ctaBtn}
          onClick={() => navigate('/generate')}
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
        >
          Start Generating
        </motion.button>
      </motion.section>
    </main>
  )
}