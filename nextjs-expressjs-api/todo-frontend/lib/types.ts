import { z } from "zod"

// Base user schema for shared fields
const BaseUserSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
})

// Extended schema for signup that includes username
export const SignupSchema = BaseUserSchema.extend({
  username: z.string().min(3).max(20)
})

// Login schema only needs email and password
export const LoginSchema = BaseUserSchema

export const TodoSchema = z.object({
  id: z.number(),
  title: z.string(),
  completed: z.boolean(),
  userId: z.number(),
  createdAt: z.string(),
  updatedAt: z.string()
})

export type LoginCredentials = z.infer<typeof LoginSchema>
export type SignupCredentials = z.infer<typeof SignupSchema>
export type Todo = z.infer<typeof TodoSchema> 