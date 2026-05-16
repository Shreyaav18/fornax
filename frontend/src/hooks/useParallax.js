import { useState, useEffect, useRef, useCallback } from 'react'

export default function useParallax({ speed = 0.3, direction = 'vertical' } = {}) {
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const frameRef = useRef(null)

  const handleScroll = useCallback(() => {
    if (frameRef.current) cancelAnimationFrame(frameRef.current)

    frameRef.current = requestAnimationFrame(() => {
      const scrollY = window.scrollY
      const scrollX = window.scrollX

      if (direction === 'vertical') {
        setOffset({ x: 0, y: scrollY * speed })
      } else if (direction === 'horizontal') {
        setOffset({ x: scrollX * speed, y: 0 })
      } else {
        setOffset({ x: scrollX * speed, y: scrollY * speed })
      }

      lastScrollY.current = scrollY
    })
  }, [speed, direction])

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()

    return () => {
      window.removeEventListener('scroll', handleScroll)
      if (frameRef.current) cancelAnimationFrame(frameRef.current)
    }
  }, [handleScroll])

  return offset
}


export function useMouseParallax({ strength = 0.02, smoothing = 0.1 } = {}) {
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const targetRef = useRef({ x: 0, y: 0 })
  const currentRef = useRef({ x: 0, y: 0 })
  const frameRef = useRef(null)
  const isMovingRef = useRef(false)

  const handleMouseMove = useCallback((e) => {
    const cx = window.innerWidth / 2
    const cy = window.innerHeight / 2
    targetRef.current = {
      x: (e.clientX - cx) * strength,
      y: (e.clientY - cy) * strength
    }
    isMovingRef.current = true
  }, [strength])

  const animate = useCallback(() => {
    if (!isMovingRef.current) {
      frameRef.current = requestAnimationFrame(animate)
      return
    }

    currentRef.current = {
      x: currentRef.current.x + (targetRef.current.x - currentRef.current.x) * smoothing,
      y: currentRef.current.y + (targetRef.current.y - currentRef.current.y) * smoothing
    }

    const dx = Math.abs(targetRef.current.x - currentRef.current.x)
    const dy = Math.abs(targetRef.current.y - currentRef.current.y)

    if (dx < 0.001 && dy < 0.001) {
      isMovingRef.current = false
    }

    setPosition({ ...currentRef.current })
    frameRef.current = requestAnimationFrame(animate)
  }, [smoothing])

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove, { passive: true })
    frameRef.current = requestAnimationFrame(animate)

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      if (frameRef.current) cancelAnimationFrame(frameRef.current)
    }
  }, [handleMouseMove, animate])

  return position
}