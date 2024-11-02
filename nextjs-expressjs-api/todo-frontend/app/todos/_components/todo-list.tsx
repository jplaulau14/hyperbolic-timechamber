'use client'

import { useEffect } from 'react'
import { useTodoStore } from '@/lib/store'
import { TodoItem } from './todo-item'
import { CreateTodo } from './create-todo'
import { Skeleton } from '@/components/ui/skeleton'

function TodoSkeleton() {
  return (
    <div className="space-y-4">
      {/* Create Todo Input Skeleton */}
      <div className="flex gap-2">
        <Skeleton className="h-10 flex-1" /> {/* Input */}
        <Skeleton className="h-10 w-16" /> {/* Add button */}
      </div>

      {/* Todo Items Skeleton */}
      <div className="space-y-2">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="flex items-center gap-2 p-4 bg-white rounded-lg shadow">
            <Skeleton className="h-4 w-4 rounded" /> {/* Checkbox */}
            <Skeleton className="h-4 flex-1" /> {/* Todo text */}
            <Skeleton className="h-8 w-8 rounded" /> {/* Delete button with icon */}
          </div>
        ))}
      </div>
    </div>
  )
}

export function TodoList() {
  const { todos, isLoading, error, fetchTodos } = useTodoStore()

  useEffect(() => {
    fetchTodos()
  }, [fetchTodos])

  if (isLoading) return <TodoSkeleton />
  if (error) return <div>Error: {error}</div>

  return (
    <div className="space-y-4">
      <CreateTodo />
      <div className="space-y-2">
        {Array.isArray(todos) && todos.map((todo) => (
          <TodoItem 
            key={`todo-${todo?.id}`}
            todo={todo}
          />
        ))}
        {(!todos || todos.length === 0) && (
          <p className="text-center text-gray-500">No todos yet. Create one above!</p>
        )}
      </div>
    </div>
  )
} 