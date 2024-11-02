import { db } from '../db';
import { todos } from '../db/schema';
import { and, eq } from 'drizzle-orm';

export class TodoService {
  async getAllTodos(userId: number) {
    return await db.select().from(todos).where(eq(todos.userId, userId));
  }

  async getTodoById(id: number, userId: number) {
    const todo = await db.select()
      .from(todos)
      .where(and(
        eq(todos.id, id),
        eq(todos.userId, userId)
      ));
    return todo[0];
  }

  async createTodo(title: string, userId: number) {
    return await db.insert(todos).values({
      title,
      userId
    }).returning();
  }

  async updateTodo(id: number, userId: number, data: { title?: string; completed?: boolean }) {
    return await db.update(todos)
      .set(data)
      .where(and(
        eq(todos.id, id),
        eq(todos.userId, userId)
      ))
      .returning();
  }

  async deleteTodo(id: number, userId: number) {
    return await db.delete(todos)
      .where(and(
        eq(todos.id, id),
        eq(todos.userId, userId)
      ));
  }
} 