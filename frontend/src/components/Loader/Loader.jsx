import { motion } from 'framer-motion'
import styles from './Loader.module.css'

const PARTICLE_COUNT = 6

const containerVariants = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
}

const particleVariants = {
  initial: { opacity: 0, scale: 0 },
  animate: (i) => ({
    opacity: [0.2, 1, 0.2],
    scale: [0.8, 1.2, 0.8],
    x: [0, Math.cos((i / PARTICLE_COUNT) * Math.PI * 2) * 24, 0],
    y: [0, Math.sin((i / PARTICLE_COUNT) * Math.PI * 2) * 24, 0],
    transition: {
      duration: 1.8,
      repeat: Infinity,
      delay: (i / PARTICLE_COUNT) * 1.8,
      ease: 'easeInOut'
    }
  })
}

export default function Loader({ size = 'md', label = 'Loading' }) {
  return (
    <div className={styles.wrapper} data-size={size}>
      <motion.div
        className={styles.orbital}
        variants={containerVariants}
        initial="initial"
        animate="animate"
      >
        {Array.from({ length: PARTICLE_COUNT }).map((_, i) => (
          <motion.span
            key={i}
            className={styles.particle}
            custom={i}
            variants={particleVariants}
            initial="initial"
            animate="animate"
          />
        ))}
        <div className={styles.core} />
      </motion.div>
      {label && (
        <motion.p
          className={styles.label}
          initial={{ opacity: 0 }}
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        >
          {label}
        </motion.p>
      )}
    </div>
  )
}