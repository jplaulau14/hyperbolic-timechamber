import { Router, Request, Response, RequestHandler } from 'express';
import { UserService } from '../services/userService';
import logger from '../utils/logger';
import { z } from 'zod';

const router = Router();
const userService = new UserService();

const userSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6)
});

router.post('/', (async (req: Request, res: Response) => {
  try {
    logger.info('Incoming request to create user', { body: req.body });
    
    const validatedData = userSchema.parse(req.body);
    
    logger.info('Creating new user', { email: validatedData.email });
    
    const user = await userService.createUser(validatedData.email, validatedData.password);
    
    logger.info('User created successfully', { userId: user[0].id });
    res.status(201).json(user);
  } catch (error) {
    logger.error('Error creating user', { 
      error: error instanceof Error ? error.message : 'Unknown error',
      stack: error instanceof Error ? error.stack : undefined,
      body: req.body 
    });

    if (error instanceof z.ZodError) {
      return res.status(400).json({ 
        message: 'Invalid input', 
        errors: error.errors 
      });
    }

    if (error instanceof Error && error.message.includes('unique constraint')) {
      return res.status(409).json({ 
        message: 'Email already exists' 
      });
    }

    res.status(500).json({ 
      message: 'Error creating user',
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}) as RequestHandler);

router.post('/login', (async (req: Request, res: Response) => {
  try {
    logger.info('Login request received:', { 
      body: {
        email: req.body.email,
        hasPassword: !!req.body.password,
      },
      headers: {
        origin: req.headers.origin,
        'content-type': req.headers['content-type'],
        'accept': req.headers.accept,
      },
      cookies: req.cookies,
      method: req.method,
      path: req.path,
    });
    
    if (!req.body.email || !req.body.password) {
      logger.warn('Missing credentials in login request');
      return res.status(400).json({ message: 'Email and password are required' });
    }
    
    const { email, password } = req.body;
    
    logger.info('Attempting login for user', { email });
    
    const token = await userService.login(email, password);
    
    logger.info('Setting cookie for successful login', { 
      email,
      cookieOptions: {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        path: '/',
      }
    });

    res.cookie('token', token, { 
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
    });
    
    logger.info('Login successful, sending response');
    res.json({ message: 'Login successful' });
  } catch (error) {
    logger.error('Login error:', { 
      error: error instanceof Error ? error.message : 'Unknown error',
      stack: error instanceof Error ? error.stack : undefined,
      email: req.body.email,
    });

    if (error instanceof Error && 
       (error.message === 'User not found' || error.message === 'Invalid password')) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    res.status(500).json({ message: 'Error during login' });
  }
}) as RequestHandler);

router.get('/validate', (async (req: Request, res: Response) => {
  try {
    const token = req.cookies.token;
    
    if (!token) {
      return res.status(401).json({ 
        valid: false, 
        message: 'No token provided' 
      });
    }

    const user = await userService.validateToken(token);
    
    res.json({ 
      valid: true, 
      user: {
        id: user.id,
        email: user.email
      }
    });
  } catch (error) {
    logger.error('Token validation error:', { 
      error: error instanceof Error ? error.message : 'Unknown error'
    });
    
    res.status(401).json({ 
      valid: false, 
      message: 'Invalid token' 
    });
  }
}) as RequestHandler);

export default router; 