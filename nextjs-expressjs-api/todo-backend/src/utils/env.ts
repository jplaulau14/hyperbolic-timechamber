import dotenv from 'dotenv';

dotenv.config();

interface Env {
  DATABASE_URL: string;
  JWT_SECRET: string;
  PORT: string;
  CORS_ORIGINS: string;
  DATABASE_USER: string;
  DATABASE_PASSWORD: string;
}

export const env: Env = {
  DATABASE_URL: process.env.DATABASE_URL || '',
  JWT_SECRET: process.env.JWT_SECRET || 'secret',
  PORT: process.env.PORT || '8080',
  CORS_ORIGINS: process.env.CORS_ORIGINS || '',
  DATABASE_USER: process.env.DATABASE_USER || '',
  DATABASE_PASSWORD: process.env.DATABASE_PASSWORD || ''
};
