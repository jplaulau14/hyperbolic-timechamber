import { LoginCredentials, SignupCredentials, Todo, User } from './types'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export const api = {
  login: async (credentials: LoginCredentials, onLoginSuccess: (user: User) => void) => {
    console.log('Attempting login:', { email: credentials.email });
    
    const res = await fetch(`${BASE_URL}/users/login`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(credentials),
      credentials: 'include'
    });

    if (!res.ok) {
      const errorData = await res.json();
      console.error('Login failed:', errorData);
      throw new Error(errorData.message || 'Login failed');
    }

    const userData = await api.validateAuth();
    onLoginSuccess(userData.user);
    return userData;
  },

  signup: async (credentials: SignupCredentials) => {
    console.log('Attempting signup:', { email: credentials.email });
    
    const res = await fetch(`${BASE_URL}/users`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(credentials),
      credentials: 'include'
    });

    if (!res.ok) {
      const errorData = await res.json();
      console.error('Signup failed:', errorData);
      throw new Error(errorData.message || 'Signup failed');
    }

    const data = await res.json();
    console.log('Signup successful:', data);
    return data;
  },

  getTodos: async () => {
    console.log('Fetching todos');
    
    const res = await fetch(`${BASE_URL}/todos`, {
      headers: {
        'Accept': 'application/json'
      },
      credentials: 'include'
    });

    if (!res.ok) {
      const errorData = await res.json();
      console.error('Failed to fetch todos:', errorData);
      throw new Error(errorData.message || 'Failed to fetch todos');
    }

    const data = await res.json();
    console.log('Todos fetched:', data);
    return data as Todo[];
  },

  createTodo: async (title: string) => {
    console.log('Creating todo:', { title });
    
    const res = await fetch(`${BASE_URL}/todos`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      credentials: 'include',
      body: JSON.stringify({ title })
    });

    if (!res.ok) {
      const errorData = await res.json();
      console.error('Failed to create todo:', errorData);
      throw new Error(errorData.message || 'Failed to create todo');
    }

    const data = await res.json();
    console.log('Todo created:', data);
    return Array.isArray(data) ? data[0] : data as Todo;
  },

  updateTodo: async (id: number, updates: Partial<Todo>) => {
    console.log('Updating todo:', { id, updates });
    
    const res = await fetch(`${BASE_URL}/todos/${id}`, {
      method: 'PUT',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      credentials: 'include',
      body: JSON.stringify(updates)
    });

    if (!res.ok) {
      const errorData = await res.json();
      console.error('Failed to update todo:', errorData);
      throw new Error(errorData.message || 'Failed to update todo');
    }

    const data = await res.json();
    console.log('Todo updated:', data);
    return data as Todo;
  },

  deleteTodo: async (id: number) => {
    console.log('Deleting todo:', { id });
    
    const res = await fetch(`${BASE_URL}/todos/${id}`, {
      method: 'DELETE',
      headers: {
        'Accept': 'application/json'
      },
      credentials: 'include'
    });

    if (!res.ok) {
      const errorData = await res.json();
      console.error('Failed to delete todo:', errorData);
      throw new Error(errorData.message || 'Failed to delete todo');
    }

    console.log('Todo deleted successfully');
  },

  validateAuth: async () => {
    console.log('Validating authentication');
    
    const res = await fetch(`${BASE_URL}/users/validate`, {
      headers: {
        'Accept': 'application/json'
      },
      credentials: 'include'
    });

    if (!res.ok) {
      throw new Error('Authentication failed');
    }

    const data = await res.json();
    console.log('Auth validation result:', data);
    return data;
  },
}; 