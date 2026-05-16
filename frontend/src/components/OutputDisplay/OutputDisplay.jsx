import { useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Copy, Check, RotateCcw } from 'lucide-react'
import { useState } from 'react'
import useTypewriter from '@hooks/useTypewriter'
import Loader from '@components/Loader/Loader'
import styles from './OutputDisplay.module.css'

export default function OutputDisplay({
  output = '',
  isLoading = false,
  error = null,
  onReset,
  typewriterSpeed = 40
}) {
  const [copied, setCopied] = useState(false)
  const scrollRef = useRef(null)

  const { displayedText, isTyping, isDone } = useTypewriter({
    text: output,
    speed: typewriterSpeed,
    startDelay: 200
  })

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [displayedText])

  const handleCopy = async () => {
    if (!output) return
    try {
      await navigator.clipboard.writeText(output)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      console.error('Clipboard write failed')
    }
  }

  const isEmpty = !output && !isLoading && !error

  return (
    <motion.div
      className={`${styles.wrapper} gradient-border`}
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1, ease: [0.25, 0.46, 0.45, 0.94] }}
    >
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.label}>Output</span>
          <AnimatePresence>
            {isTyping && (
              <motion.span
                className={styles.typingBadge}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                transition={{ duration: 0.2 }}
              >
                generating
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        <div className={styles.headerActions}>
          <AnimatePresence>
            {isDone && output && (
              <motion.button
                className={styles.actionBtn}
                onClick={handleCopy}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                whileTap={{ scale: 0.9 }}
                title="Copy output"
              >
                <AnimatePresence mode="wait">
                  {copied ? (
                    <motion.span
                      key="check"
                      initial={{ opacity: 0, scale: 0.7 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.7 }}
                    >
                      <Check size={14} />
                    </motion.span>
                  ) : (
                    <motion.span
                      key="copy"
                      initial={{ opacity: 0, scale: 0.7 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.7 }}
                    >
                      <Copy size={14} />
                    </motion.span>
                  )}
                </AnimatePresence>
              </motion.button>
            )}
          </AnimatePresence>

          {onReset && (
            <motion.button
              className={styles.actionBtn}
              onClick={onReset}
              whileTap={{ scale: 0.9 }}
              title="Reset"
            >
              <RotateCcw size={14} />
            </motion.button>
          )}
        </div>
      </div>

      <div className={styles.body} ref={scrollRef}>
        <AnimatePresence mode="wait">
          {isLoading && !output && (
            <motion.div
              key="loader"
              className={styles.loaderWrapper}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Loader size="md" label="Generating" />
            </motion.div>
          )}

          {error && (
            <motion.div
              key="error"
              className={styles.errorWrapper}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <span className={styles.errorText}>{error}</span>
            </motion.div>
          )}

          {isEmpty && !error && (
            <motion.div
              key="empty"
              className={styles.emptyWrapper}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <span className={styles.emptyText}>
                Output will appear here
              </span>
            </motion.div>
          )}

          {output && !error && (
            <motion.div
              key="output"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <p className={styles.outputText}>
                {displayedText}
                {isTyping && (
                  <motion.span
                    className={styles.cursor}
                    animate={{ opacity: [1, 0, 1] }}
                    transition={{ duration: 0.8, repeat: Infinity }}
                  />
                )}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {output && isDone && (
        <motion.div
          className={styles.footer}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
        >
          <span className={styles.tokenCount}>
            {output.split(' ').length} words generated
          </span>
        </motion.div>
      )}
    </motion.div>
  )
}