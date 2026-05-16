import { motion } from 'framer-motion'
import styles from './PageTransition.module.css'

const variants = {
  initial: {
    opacity: 0,
    y: 24,
    filter: 'blur(8px)'
  },
  animate: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: {
      duration: 0.5,
      ease: [0.25, 0.46, 0.45, 0.94]
    }
  },
  exit: {
    opacity: 0,
    y: -16,
    filter: 'blur(8px)',
    transition: {
      duration: 0.3,
      ease: [0.25, 0.46, 0.45, 0.94]
    }
  }
}

export default function PageTransition({ children }) {
  return (
    <motion.div
      className={styles.wrapper}
      variants={variants}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      {children}
    </motion.div>
  )
}