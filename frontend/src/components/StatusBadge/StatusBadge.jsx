import { motion } from 'framer-motion'
import styles from './StatusBadge.module.css'

const STATUS_CONFIG = {
  completed: { label: 'Completed', pulse: false },
  running: { label: 'Running', pulse: true },
  failed: { label: 'Failed', pulse: false },
  pending: { label: 'Pending', pulse: true }
}

export default function StatusBadge({ status = 'pending' }) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.pending

  return (
    <motion.div
      className={styles.badge}
      data-status={status}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      <span className={styles.dot} data-status={status}>
        {config.pulse && (
          <motion.span
            className={styles.pulse}
            animate={{ scale: [1, 2, 1], opacity: [0.8, 0, 0.8] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
          />
        )}
      </span>
      <span className={styles.label}>{config.label}</span>
    </motion.div>
  )
}