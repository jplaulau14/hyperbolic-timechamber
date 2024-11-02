import { TodoList } from './_components/todo-list'
import { AuthCheck } from '@/components/auth-check'

export default function TodosPage() {
  return (
    <AuthCheck>
      <div className="max-w-2xl mx-auto p-4">
        <h1 className="text-2xl font-bold mb-6">My Todos</h1>
        <TodoList />
      </div>
    </AuthCheck>
  )
} 