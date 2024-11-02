import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { Todo } from './types'
import { api } from './api'

interface TodoStore {
  todos: Todo[]
  isLoading: boolean
  error: string | null
  fetchTodos: () => Promise<void>
  addTodo: (title: string) => Promise<void>
  toggleTodo: (id: number) => Promise<void>
  deleteTodo: (id: number) => Promise<void>
}

export const useTodoStore = create<TodoStore>((set, get) => ({
  todos: [],
  isLoading: true,
  error: null,
  fetchTodos: async () => {
    set({ isLoading: true })
    try {
      await new Promise(resolve => setTimeout(resolve, 2000))
      const todos = await api.getTodos()
      set({ todos, error: null })
    } catch (error) {
      set({ error: (error as Error).message })
    } finally {
      set({ isLoading: false })
    }
  },
  addTodo: async (title: string) => {
    try {
      const newTodo = await api.createTodo(title)
      set(state => ({ todos: [...state.todos, newTodo] }))
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },
  toggleTodo: async (id: number) => {
    try {
      const todo = get().todos.find(t => t.id === id)
      if (!todo) return
      const updated = await api.updateTodo(id, { completed: !todo.completed })
      set(state => ({
        todos: state.todos.map(t => t.id === id ? updated : t)
      }))
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },
  deleteTodo: async (id: number) => {
    try {
      await api.deleteTodo(id)
      set(state => ({
        todos: state.todos.filter(t => t.id !== id)
      }))
    } catch (error) {
      set({ error: (error as Error).message })
    }
  }
}))

interface User {
  id: number;
  email: string;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  expiresAt: number | null;
}

interface AuthStore extends AuthState {
  validateAuth: () => Promise<void>;
  setUser: (user: User | null) => void;
  logout: () => void;
  checkAuthExpiry: () => boolean;
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,
      isLoading: true,
      error: null,
      expiresAt: null,

      validateAuth: async () => {
        if (get().checkAuthExpiry()) {
          set({ isLoading: false });
          return;
        }

        set({ isLoading: true });
        try {
          const { valid, user } = await api.validateAuth();
          const expiresAt = Date.now() + (12 * 60 * 60 * 1000);
          
          set({ 
            isAuthenticated: valid, 
            user: valid ? user : null,
            error: null,
            expiresAt,
            isLoading: false
          });
        } catch (error) {
          set({ 
            isAuthenticated: false, 
            user: null,
            error: (error as Error).message,
            expiresAt: null,
            isLoading: false
          });
        }
      },

      setUser: (user) => {
        const expiresAt = Date.now() + (12 * 60 * 60 * 1000);
        set({ 
          user, 
          isAuthenticated: !!user,
          expiresAt
        });
      },

      logout: () => {
        set({ 
          user: null, 
          isAuthenticated: false,
          expiresAt: null
        });
      },

      checkAuthExpiry: () => {
        const state = get();
        const isValid = Boolean(
          state.isAuthenticated && 
          state.user && 
          state.expiresAt && 
          Date.now() < state.expiresAt
        );
        
        if (!isValid && state.isAuthenticated) {
          get().logout();
        }
        
        return isValid;
      }
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ 
        user: state.user,
        isAuthenticated: state.isAuthenticated,
        expiresAt: state.expiresAt
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
          state.validateAuth();
        }
      }
    }
  )
); 