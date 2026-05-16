import { useState, useEffect, useRef, useCallback } from 'react'

export default function useTypewriter({
  text = '',
  speed = 30,
  startDelay = 0,
  onComplete = null
} = {}) {
  const [displayedText, setDisplayedText] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [isDone, setIsDone] = useState(false)
  const indexRef = useRef(0)
  const timeoutRef = useRef(null)
  const startDelayRef = useRef(null)

  const clearTimers = useCallback(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    if (startDelayRef.current) clearTimeout(startDelayRef.current)
  }, [])

  const reset = useCallback(() => {
    clearTimers()
    setDisplayedText('')
    setIsTyping(false)
    setIsDone(false)
    indexRef.current = 0
  }, [clearTimers])

  useEffect(() => {
    if (!text) {
      reset()
      return
    }

    reset()
    setIsTyping(true)

    startDelayRef.current = setTimeout(() => {
      const type = () => {
        if (indexRef.current < text.length) {
          const charsPerTick = speed > 60 ? 3 : 1
          const nextIndex = Math.min(indexRef.current + charsPerTick, text.length)
          setDisplayedText(text.slice(0, nextIndex))
          indexRef.current = nextIndex
          timeoutRef.current = setTimeout(type, 1000 / speed)
        } else {
          setIsTyping(false)
          setIsDone(true)
          if (onComplete) onComplete()
        }
      }
      type()
    }, startDelay)

    return clearTimers
  }, [text, speed, startDelay, onComplete, clearTimers, reset])

  return { displayedText, isTyping, isDone, reset }
}