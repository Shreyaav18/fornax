import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { useParams, useNavigate } from 'react-router-dom'
import { X, ArrowLeft } from 'lucide-react'
import RunCard from '@components/RunCard/RunCard'
import StatusBadge from '@components/StatusBadge/StatusBadge'
import Loader from '@components/Loader/Loader'
import { fetchRuns, fetchRun } from '@api/runs'
import styles from './Runs.module.css'

function RunDetail({ runId, onClose }) {
  const { data: run, isLoading, error } = useQuery({
    queryKey: ['run', runId],
    queryFn: () => fetchRun(runId),
    enabled: !!runId
  })

  return (
    <motion.div
      className={styles.detailOverlay}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      onClick={onClose}
    >
      <motion.div
        className={`${styles.detailPanel} glass`}
        initial={{ x: '100%', opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: '100%', opacity: 0 }}
        transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.detailHeader}>
          <button className={styles.closeBtn} onClick={onClose}>
            <X size={18} />
          </button>
          {run && <StatusBadge status={run.status} />}
        </div>

        {isLoading && (
          <div className={styles.detailLoader}>
            <Loader size="md" label="Loading run" />
          </div>
        )}

        {error && (
          <div className={styles.detailError}>
            <p>Failed to load run details</p>
          </div>
        )}

        {run && !isLoading && (
          <div className={styles.detailContent}>
            <div className={styles.detailTop}>
              <h2 className={styles.detailName}>{run.name}</h2>
              <span className={styles.detailId}>Run #{run.id}</span>
            </div>

            <div className={styles.detailSection}>
              <span className={styles.detailSectionLabel}>Training</span>
              <div className={styles.detailGrid}>
                {[
                  { label: 'Best Val Loss', value: run.best_val_loss?.toFixed(4) ?? '—' },
                  { label: 'Final Step', value: run.final_step ?? '—' },
                  { label: 'Started', value: run.started_at ? new Date(run.started_at).toLocaleString() : '—' },
                  { label: 'Finished', value: run.finished_at ? new Date(run.finished_at).toLocaleString() : '—' }
                ].map((item) => (
                  <div key={item.label} className={styles.detailStat}>
                    <span className={styles.detailStatLabel}>{item.label}</span>
                    <span className={styles.detailStatValue}>{item.value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className={styles.detailSection}>
              <span className={styles.detailSectionLabel}>Model Config</span>
              <div className={styles.configBlock}>
                {Object.entries(run.model_config ?? {}).map(([key, val]) => (
                  <div key={key} className={styles.configRow}>
                    <span className={styles.configKey}>{key}</span>
                    <span className={styles.configVal}>{String(val)}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className={styles.detailSection}>
              <span className={styles.detailSectionLabel}>Train Config</span>
              <div className={styles.configBlock}>
                {Object.entries(run.train_config ?? {}).map(([key, val]) => (
                  <div key={key} className={styles.configRow}>
                    <span className={styles.configKey}>{key}</span>
                    <span className={styles.configVal}>{String(val)}</span>
                  </div>
                ))}
              </div>
            </div>

            {run.notes && (
              <div className={styles.detailSection}>
                <span className={styles.detailSectionLabel}>Notes</span>
                <p className={styles.detailNotes}>{run.notes}</p>
              </div>
            )}
          </div>
        )}
      </motion.div>
    </motion.div>
  )
}

export default function Runs() {
  const [selectedRunId, setSelectedRunId] = useState(null)
  const navigate = useNavigate()

  const { data: runs = [], isLoading, error } = useQuery({
    queryKey: ['runs'],
    queryFn: fetchRuns,
    refetchInterval: 10000
  })

  return (
    <main className={styles.main}>
      <div className={styles.bg}>
        <motion.div
          className={styles.bgBlob}
          animate={{ scale: [1, 1.1, 1], opacity: [0.5, 0.8, 0.5] }}
          transition={{ duration: 16, repeat: Infinity, ease: 'easeInOut' }}
        />
      </div>

      <div className={styles.content}>
        <motion.div
          className={styles.header}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
        >
          <div className={styles.headerTop}>
            <h1 className={`${styles.title} gradient-text`}>Training Runs</h1>
            <motion.div
              className={styles.countBadge}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.2 }}
            >
              <span className={styles.countText}>{runs.length}</span>
            </motion.div>
          </div>
          <p className={styles.subtitle}>
            All training runs — click any card to inspect
          </p>
        </motion.div>

        {isLoading && (
          <div className={styles.loaderWrapper}>
            <Loader size="lg" label="Loading runs" />
          </div>
        )}

        {error && (
          <motion.div
            className={styles.errorWrapper}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <p className={styles.errorText}>Failed to load runs</p>
          </motion.div>
        )}

        {!isLoading && !error && runs.length === 0 && (
          <motion.div
            className={styles.emptyWrapper}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <p className={styles.emptyText}>No training runs yet</p>
            <p className={styles.emptySubtext}>
              Run train.py to start your first training run
            </p>
          </motion.div>
        )}

        {!isLoading && !error && runs.length > 0 && (
          <motion.div
            className={styles.grid}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.2 }}
          >
            {runs.map((run, i) => (
              <div
                key={run.id}
                onClick={() => setSelectedRunId(run.id)}
              >
                <RunCard run={run} index={i} />
              </div>
            ))}
          </motion.div>
        )}
      </div>

      <AnimatePresence>
        {selectedRunId && (
          <RunDetail
            runId={selectedRunId}
            onClose={() => setSelectedRunId(null)}
          />
        )}
      </AnimatePresence>
    </main>
  )
}