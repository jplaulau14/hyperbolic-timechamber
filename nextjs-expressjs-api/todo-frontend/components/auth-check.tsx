'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '../lib/store'

export function AuthCheck({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const { isAuthenticated, isLoading, validateAuth, checkAuthExpiry } = useAuthStore()
  const [isInitializing, setIsInitializing] = useState(true)

  useEffect(() => {
    const initAuth = async () => {
      try {
        if (!checkAuthExpiry()) {
          await validateAuth()
        }
      } finally {
        setIsInitializing(false)
      }
    }

    initAuth()
  }, [validateAuth, checkAuthExpiry])

  useEffect(() => {
    if (!isInitializing && !isLoading && !isAuthenticated) {
      router.push('/login')
    }
  }, [isInitializing, isLoading, isAuthenticated, router])

  return <>{children}</>
} 