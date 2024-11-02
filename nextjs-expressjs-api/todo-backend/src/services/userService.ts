import { db } from '../db';
import { users } from '../db/schema';
import { eq } from 'drizzle-orm';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import logger from '../utils/logger';

export class UserService {
  async createUser(email: string, password: string) {
    try {
      if (typeof password !== 'string') {
        throw new Error('Password must be a string');
      }

      const hashedPassword = await bcrypt.hash(password, 10);
      
      logger.debug('Attempting to create user', { email });
      
      const result = await db.insert(users).values({
        email,
        password: hashedPassword
      }).returning();
      
      logger.info('User created successfully', { userId: result[0].id });
      
      return result;
    } catch (error) {
      logger.error('Database error in createUser', {
        error: error instanceof Error ? error.message : 'Unknown error',
        email
      });
      throw error; 
    }
  }

  async login(email: string, password: string) {
    try {
      const user = await db.select().from(users).where(eq(users.email, email)).limit(1);
      
      if (!user.length) {
        logger.warn('Login attempt with non-existent user', { email });
        throw new Error('User not found');
      }

      const validPassword = await bcrypt.compare(password, user[0].password);
      if (!validPassword) {
        logger.warn('Invalid password attempt', { email });
        throw new Error('Invalid password');
      }

      const token = jwt.sign(
        { userId: user[0].id }, 
        process.env.JWT_SECRET || 'secret',
        { expiresIn: '24h' }
      );

      logger.info('User logged in successfully', { userId: user[0].id });
      return token;
    } catch (error) {
      logger.error('Error in login process', {
        error: error instanceof Error ? error.message : 'Unknown error',
        email
      });
      throw error;
    }
  }

  async validateToken(token: string) {
    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: number };
      
      const user = await db.select()
        .from(users)
        .where(eq(users.id, decoded.userId))
        .limit(1);
      
      if (!user.length) {
        throw new Error('User not found');
      }
      
      return user[0];
    } catch (error) {
      logger.error('Token validation error:', {
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      throw new Error('Invalid token');
    }
  }
} 