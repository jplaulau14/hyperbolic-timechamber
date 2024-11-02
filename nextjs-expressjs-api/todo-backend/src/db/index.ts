import { drizzle } from 'drizzle-orm/node-postgres';
import { Pool } from 'pg';
import * as schema from './schema';
import logger from '../utils/logger';
import { env } from '../utils/env';

const pool = new Pool({
  connectionString: env.DATABASE_URL,
  ssl: false,
  password: env.DATABASE_PASSWORD,
  user: env.DATABASE_USER
});

// Enhanced connection testing
const testConnection = async () => {
  try {
    const client = await pool.connect();
    
    // Log database connection details
    const result = await client.query(`
      SELECT current_database() as database,
             current_user as user,
             current_schema as schema;
    `);
    
    // Log table information
    const tables = await client.query(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public';
    `);

    logger.info('Database Connection Details:', {
      database: result.rows[0],
      tables: tables.rows.map(row => row.table_name)
    });

    client.release();
    return true;
  } catch (err) {
    logger.error('Database Connection Error:', {
      error: err instanceof Error ? err.message : 'Unknown error',
      connectionString: env.DATABASE_URL.replace(/:([^:@]{8})[^:@]*@/, ':****@') // Mask password in logs
    });
    return false;
  }
};

// Test the connection immediately
testConnection()
  .then((success) => {
    if (!success) {
      logger.error('Failed to establish database connection');
      process.exit(1);
    }
  });

export const db = drizzle(pool, { schema });