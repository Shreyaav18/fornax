import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Activity, Layers, TrendingDown, Clock } from 'lucide-react'
import StatusBadge from '@components/StatusBadge/StatusBadge'
import styles from './RunCard.module.css'

function formatDate(dateStr) {
  if (!dateStr) return '—'
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(new Date(dateStr))
}

function formatLoss(loss) {
  if (loss === null || loss === undefined) return '—'
  return loss.toFixed(4)
}

export default function RunCard({ run, index = 0 }) {
  const navigate = useNavigate()

  const handleClick = () => {
    navigate(`/runs/${run.id}`)
  }

  return (
    <motion.div
      className={`${styles.card} gradient-border`}
      initial={{ opacity: 0, y: 32 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.5,
        delay: index * 0.08,
        ease: [0.25, 0.46, 0.45, 0.94]
      }}
      whileHover={{
        y: -6,
        transition: { duration: 0.2, ease: 'easeOut' }
      }}
      onClick={handleClick}
    >
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h3 className={styles.name}>{run.name}</h3>
          <span className={styles.id}>#{run.id}</span>
        </div>
        <StatusBadge status={run.status} />
      </div>

      <div className={styles.grid}>
        <div className={styles.stat}>
          <TrendingDown size={14} className={styles.statIcon} />
          <span className={styles.statLabel}>Best Loss</span>
          <span className={styles.statValue}>{formatLoss(run.best_val_loss)}</span>
        </div>

        <div className={styles.stat}>
          <Activity size={14} className={styles.statIcon} />
          <span className={styles.statLabel}>Steps</span>
          <span className={styles.statValue}>{run.final_step ?? '—'}</span>
        </div>

        <div className={styles.stat}>
          <Layers size={14} className={styles.statIcon} />
          <span className={styles.statLabel}>Layers</span>
          <span className={styles.statValue}>{run.model_config?.n_layers ?? '—'}</span>
        </div>

        <div className={styles.stat}>
          <Clock size={14} className={styles.statIcon} />
          <span className={styles.statLabel}>Started</span>
          <span className={styles.statValue}>{formatDate(run.started_at)}</span>
        </div>
      </div>

      <div className={styles.footer}>
        <span className={styles.footerLabel}>
          d_model {run.model_config?.d_model ?? '—'} &nbsp;·&nbsp;
          {run.model_config?.n_heads ?? '—'} heads &nbsp;·&nbsp;
          vocab {run.model_config?.vocab_size ?? '—'}
        </span>
        <motion.span
          className={styles.cta}
          initial={{ x: 0 }}
          whileHover={{ x: 4 }}
          transition={{ duration: 0.2 }}
        >
          View run →
        </motion.span>
      </div>
    </motion.div>
  )
}