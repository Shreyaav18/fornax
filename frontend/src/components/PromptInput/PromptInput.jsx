import { useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowUp, X } from 'lucide-react'
import styles from './PromptInput.module.css'

export default function PromptInput({
  value = '',
  onChange,
  onSubmit,
  isLoading = false,
  disabled = false,
  placeholder = 'Enter a prompt...',
  maxLength = 512
}) {
  const textareaRef = useRef(null)
  const [isFocused, setIsFocused] = useState(false)

  const handleChange = useCallback((e) => {
    if (onChange) onChange(e.target.value)
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = `${Math.min(el.scrollHeight, 200)}px`
    }
  }, [onChange])

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!disabled && !isLoading && value.trim()) {
        onSubmit?.()
      }
    }
  }, [disabled, isLoading, value, onSubmit])

  const handleClear = useCallback(() => {
    if (onChange) onChange('')
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.focus()
    }
  }, [onChange])

  const charPercent = (value.length / maxLength) * 100
  const isNearLimit = charPercent > 80
  const isAtLimit = value.length >= maxLength
  const canSubmit = value.trim().length > 0 && !disabled && !isLoading

  return (
    <motion.div
      className={styles.wrapper}
      data-focused={isFocused}
      data-loading={isLoading}
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
    >
      <motion.div
        className={styles.glowRing}
        animate={{
          opacity: isFocused ? 1 : 0,
          scale: isFocused ? 1 : 0.98
        }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
      />

      <div className={styles.inner}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          disabled={disabled || isLoading}
          maxLength={maxLength}
          rows={1}
          spellCheck={false}
        />

        <div className={styles.actions}>
          <AnimatePresence>
            {value.length > 0 && (
              <motion.button
                className={styles.clearBtn}
                onClick={handleClear}
                initial={{ opacity: 0, scale: 0.7 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.7 }}
                transition={{ duration: 0.15 }}
                whileTap={{ scale: 0.9 }}
                type="button"
              >
                <X size={14} />
              </motion.button>
            )}
          </AnimatePresence>

          <motion.button
            className={styles.submitBtn}
            onClick={() => canSubmit && onSubmit?.()}
            disabled={!canSubmit}
            data-active={canSubmit}
            whileTap={canSubmit ? { scale: 0.92 } : {}}
            type="button"
          >
            <motion.div
              animate={isLoading ? {
                rotate: 360
              } : { rotate: 0 }}
              transition={isLoading ? {
                duration: 1.2,
                repeat: Infinity,
                ease: 'linear'
              } : {}}
            >
              <ArrowUp size={16} />
            </motion.div>
          </motion.button>
        </div>
      </div>

      <div className={styles.footer}>
        <span className={styles.hint}>
          Shift + Enter for new line
        </span>

        <motion.span
          className={styles.charCount}
          data-near-limit={isNearLimit}
          data-at-limit={isAtLimit}
          animate={{ opacity: isFocused ? 1 : 0.4 }}
        >
          {value.length} / {maxLength}
        </motion.span>
      </div>

      <AnimatePresence>
        {isFocused && (
          <motion.div
            className={styles.scanLine}
            initial={{ scaleX: 0, opacity: 0 }}
            animate={{ scaleX: 1, opacity: 1 }}
            exit={{ scaleX: 0, opacity: 0 }}
            transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
          />
        )}
      </AnimatePresence>
    </motion.div>
  )
}