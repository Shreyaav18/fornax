import { useRef, useCallback, useEffect } from 'react'

export default function useMagneticHover({ strength = 0.3, smoothing = 0.15 } = {}) {
  const elementRef = useRef(null)
  const frameRef = useRef(null)
  const targetRef = useRef({ x: 0, y: 0 })
  const currentRef = useRef({ x: 0, y: 0 })
  const isHoveringRef = useRef(false)
  const EPSILON = 0.001

  const animate = useCallback(() => {
    const el = elementRef.current
    if (!el) {
      frameRef.current = null
      return
    }

    currentRef.current = {
      x: currentRef.current.x + (targetRef.current.x - currentRef.current.x) * smoothing,
      y: currentRef.current.y + (targetRef.current.y - currentRef.current.y) * smoothing
    }

    const dx = Math.abs(targetRef.current.x - currentRef.current.x)
    const dy = Math.abs(targetRef.current.y - currentRef.current.y)

    el.style.transform = `translate(${currentRef.current.x}px, ${currentRef.current.y}px)`

    if (!isHoveringRef.current && dx < EPSILON && dy < EPSILON) {
      frameRef.current = null
      return
    }

    frameRef.current = requestAnimationFrame(animate)
  }, [smoothing])

  const handleMouseMove = useCallback((e) => {
    const el = elementRef.current
    if (!el) return

    const rect = el.getBoundingClientRect()
    const cx = rect.left + rect.width / 2
    const cy = rect.top + rect.height / 2

    targetRef.current = {
      x: (e.clientX - cx) * strength,
      y: (e.clientY - cy) * strength
    }
  }, [strength])

  const handleMouseEnter = useCallback(() => {
    isHoveringRef.current = true
    if (!frameRef.current) {
      frameRef.current = requestAnimationFrame(animate)
    }
  }, [animate])

  const handleMouseLeave = useCallback(() => {
    isHoveringRef.current = false
    targetRef.current = { x: 0, y: 0 }
    if (!frameRef.current) {
      frameRef.current = requestAnimationFrame(animate)
    }
  }, [animate])

  useEffect(() => {
    const el = elementRef.current
    if (!el) return

    el.addEventListener('mousemove', handleMouseMove, { passive: true })
    el.addEventListener('mouseenter', handleMouseEnter)
    el.addEventListener('mouseleave', handleMouseLeave)

    return () => {
      el.removeEventListener('mousemove', handleMouseMove)
      el.removeEventListener('mouseenter', handleMouseEnter)
      el.removeEventListener('mouseleave', handleMouseLeave)
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current)
        frameRef.current = null
      }
    }
  }, [handleMouseMove, handleMouseEnter, handleMouseLeave, animate])

  return elementRef
}