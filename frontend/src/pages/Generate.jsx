import { useCallback } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import PromptInput from '@components/PromptInput/PromptInput'
import OutputDisplay from '@components/OutputDisplay/OutputDisplay'
import Loader from '@components/Loader/Loader'
import styles from './Generate.module.css'
import useAppStore from '@store/useAppStore'
import { generateText } from '@api/generate'
import { fetchRuns } from '@api/runs'

export default function Generate() {
  const {
    prompt,
    setPrompt,
    generatedOutput,
    isGenerating,
    generationError,
    currentRun,
    setCurrentRun,
    startGeneration,
    completeGeneration,
    setGenerationError,
    clearOutput
  } = useAppStore()

  const { data: runs = [], isLoading: runsLoading } = useQuery({
    queryKey: ['runs'],
    queryFn: fetchRuns
  })

  const completedRuns = runs.filter((r) => r.status === 'completed')

  const handleSubmit = useCallback(async () => {
    if (!prompt.trim() || !currentRun) return
    startGeneration()
    try {
      const result = await generateText({
        runId: currentRun.id,
        prompt
      })
      completeGeneration(result.output)
    } catch (err) {
      setGenerationError(err.message || 'Generation failed')
    }
  }, [prompt, currentRun, startGeneration, completeGeneration, setGenerationError])

  return (
    <main className={styles.main}>
      <div className={styles.bg}>
        <motion.div
          className={styles.bgBlob1}
          animate={{ x: [0, 30, 0], y: [0, -20, 0] }}
          transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
        />
        <motion.div
          className={styles.bgBlob2}
          animate={{ x: [0, -20, 0], y: [0, 30, 0] }}
          transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut', delay: 2 }}
        />
      </div>

      <div className={styles.content}>
        <motion.div
          className={styles.header}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
        >
          <h1 className={`${styles.title} gradient-text`}>Generate</h1>
          <p className={styles.subtitle}>
            Prompt Fornax and watch it think
          </p>
        </motion.div>

        <motion.div
          className={styles.runSelector}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          {runsLoading ? (
            <Loader size="sm" label="Loading runs" />
          ) : completedRuns.length === 0 ? (
            <p className={styles.noRuns}>
              No completed runs found — train a model first
            </p>
          ) : (
            <div className={styles.runList}>
              <span className={styles.runListLabel}>Select a run</span>
              <div className={styles.runChips}>
                {completedRuns.map((run) => (
                  <motion.button
                    key={run.id}
                    className={styles.runChip}
                    data-active={currentRun?.id === run.id}
                    onClick={() => setCurrentRun(run)}
                    whileTap={{ scale: 0.95 }}
                    transition={{ duration: 0.15 }}
                  >
                    <span className={styles.chipName}>{run.name}</span>
                    <span className={styles.chipId}>#{run.id}</span>
                    <span className={styles.chipLoss}>
                      {run.best_val_loss?.toFixed(3) ?? '—'}
                    </span>
                  </motion.button>
                ))}
              </div>
            </div>
          )}
        </motion.div>

        <div className={styles.interface}>
          <OutputDisplay
            output={generatedOutput}
            isLoading={isGenerating}
            error={generationError}
            onReset={clearOutput}
          />
          <PromptInput
            value={prompt}
            onChange={setPrompt}
            onSubmit={handleSubmit}
            isLoading={isGenerating}
            disabled={!currentRun}
            placeholder={
              currentRun
                ? `Prompt ${currentRun.name}...`
                : 'Select a run above to start generating'
            }
          />
        </div>
      </div>
    </main>
  )
}