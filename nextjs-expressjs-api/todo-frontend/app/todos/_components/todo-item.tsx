'use client'

import { Todo } from '@/lib/types'
import { useTodoStore } from '@/lib/store'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Trash2 } from 'lucide-react'

interface TodoItemProps {
  todo: Todo
}

export function TodoItem({ todo }: TodoItemProps) {
  const { toggleTodo, deleteTodo } = useTodoStore()

  return (
    <div className="flex items-center gap-2 p-4 bg-white rounded-lg shadow">
      <Checkbox
        checked={todo.completed}
        onCheckedChange={() => toggleTodo(todo.id)}
      />
      <span className={`flex-1 ${todo.completed ? 'line-through text-gray-500' : ''}`}>
        {todo.title}
      </span>
      <Button
        variant="destructive"
        size="sm"
        onClick={() => deleteTodo(todo.id)}
        className="px-2 h-8 w-8"
      >
        <Trash2 className="h-4 w-4" />
      </Button>
    </div>
  )
} 