import { Router, RequestHandler } from 'express';
import { TodoService } from '../services/todoService';
import { auth, AuthRequest } from '../middleware/auth';

const router = Router();
const todoService = new TodoService();

router.use(auth as RequestHandler);

router.get('/', (async (req: AuthRequest, res) => {
  const todos = await todoService.getAllTodos(req.userId!);
  res.json(todos);
}) as RequestHandler);

router.get('/:id', (async (req: AuthRequest, res) => {
  const todo = await todoService.getTodoById(parseInt(req.params.id), req.userId!);
  if (!todo) return res.status(404).json({ message: 'Todo not found' });
  res.json(todo);
}) as RequestHandler);

router.post('/', (async (req: AuthRequest, res) => {
  const { title } = req.body;
  const todo = await todoService.createTodo(title, req.userId!);
  res.status(201).json(todo);
}) as RequestHandler);

router.put('/:id', (async (req: AuthRequest, res) => {
  const todo = await todoService.updateTodo(parseInt(req.params.id), req.userId!, req.body);
  if (!todo.length) return res.status(404).json({ message: 'Todo not found' });
  res.json(todo[0]);
}) as RequestHandler);

router.delete('/:id', (async (req: AuthRequest, res) => {
  await todoService.deleteTodo(parseInt(req.params.id), req.userId!);
  res.status(204).send();
}) as RequestHandler);

export default router; 