import express from 'express';
import cookieParser from 'cookie-parser';
import dotenv from 'dotenv';
import cors from 'cors';
import userRoutes from './api/users';
import todoRoutes from './api/todos';
import { env } from './utils/env';
import logger from './utils/logger';

dotenv.config();

const app = express();
const port = env.PORT;

app.use((req, res, next) => {
  logger.info('Incoming request:', {
    method: req.method,
    path: req.path,
    body: req.body,
    headers: {
      origin: req.headers.origin,
      'content-type': req.headers['content-type'],
      cookie: req.headers.cookie,
    },
    query: req.query,
    params: req.params,
  });
  next();
});

const corsOptions = {
  origin: function (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) {
    const allowedOrigins = env.CORS_ORIGINS.split(',');
    
    if (!origin) {
      logger.info('Request with no origin allowed', { 
        allowedOrigins,
        headers: 'No origin header'
      });
      return callback(null, true);
    }
    
    if (allowedOrigins.indexOf(origin) !== -1) {
      logger.info('CORS request details:', { 
        origin,
        allowed: true,
        allowedOrigins
      });
      callback(null, true);
    } else {
      logger.warn('CORS blocked:', { 
        origin,
        allowedOrigins,
        allowed: false
      });
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin', 'Cookie'],
  exposedHeaders: ['Set-Cookie'],
};

app.use(cors(corsOptions));
app.options('*', cors(corsOptions));
app.use(express.json());
app.use(cookieParser());

app.get('/routes', (req, res) => {
  const routes: string[] = [];
  
  app._router.stack.forEach((middleware: any) => {
    if (middleware.route) {
      routes.push(`${Object.keys(middleware.route.methods)} ${middleware.route.path}`);
    } else if (middleware.name === 'router') {
      middleware.handle.stack.forEach((handler: any) => {
        if (handler.route) {
          const path = handler.route.path;
          const methods = Object.keys(handler.route.methods);
          routes.push(`${methods.join(',')} /users${path}`);
        }
      });
    }
  });

  res.json({
    availableRoutes: routes,
    note: 'These are all available API endpoints'
  });
});

app.get('/', (req, res) => {
  res.json({
    message: 'API is running',
    endpoints: {
      auth: {
        register: 'POST /users',
        login: 'POST /users/login'
      },
      todos: {
        list: 'GET /todos',
        create: 'POST /todos',
        get: 'GET /todos/:id',
        update: 'PUT /todos/:id',
        delete: 'DELETE /todos/:id'
      }
    }
  });
});

app.use('/users', userRoutes);
app.use('/todos', todoRoutes);

app.get('/cors-test', (req, res) => {
  res.json({
    message: 'CORS test successful',
    origin: req.headers.origin,
    headers: req.headers,
  });
});

app.use((req, res) => {
  logger.warn('Route not found:', {
    method: req.method,
    path: req.path,
    origin: req.headers.origin
  });
  res.status(404).json({ 
    message: 'Route not found', 
    suggestion: 'Did you mean /users/login?',
    availableRoutes: {
      auth: {
        register: 'POST /users',
        login: 'POST /users/login'
      },
      todos: {
        list: 'GET /todos',
        create: 'POST /todos',
        get: 'GET /todos/:id',
        update: 'PUT /todos/:id',
        delete: 'DELETE /todos/:id'
      }
    }
  });
});

app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Unhandled error:', { error: err.message, stack: err.stack });
  res.status(500).json({ error: 'Internal Server Error' });
});

app.listen(port, () => {
  logger.info(`Server running on port ${port}`, {
    allowedOrigins: env.CORS_ORIGINS.split(','),
    availableEndpoints: [
      'POST /users',
      'POST /users/login',
      'GET /todos',
      'POST /todos',
      'GET /todos/:id',
      'PUT /todos/:id',
      'DELETE /todos/:id'
    ]
  });
}); 