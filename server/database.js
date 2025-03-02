import pkg from "pg";
import dotenv from "dotenv";

const { Client } = pkg;
dotenv.config();

/**
 * Creates a new PostgreSQL client without connecting to a specific database.
 * Used for checking and creating databases.
 */
function getAdminClient() {
  return new Client({
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
  });
}

/**
 * Checks if a PostgreSQL database exists.
 * @returns {Promise<boolean>} True if DB exists, false otherwise.
 */
export async function checkDatabase() {
  const client = getAdminClient();

  try {
    await client.connect();

    const res = await client.query(
      "SELECT 1 FROM pg_database WHERE datname = $1",
      [process.env.DB_NAME]
    );

    return res.rowCount > 0;
  } catch (error) {
    console.error("❌ Error checking database existence:", error);
    return false;
  } finally {
    await client.end();
  }
}

/**
 * Creates a new PostgreSQL database if it does not exist.
 * @returns {Promise<boolean>} True if created, false if already exists.
 */
export async function createDatabase() {
  const adminClient = getAdminClient();

  try {
    await adminClient.connect();

    // Check if the database already exists
    const dbName = process.env.DB_NAME;
    const checkDbQuery = `SELECT 1 FROM pg_database WHERE datname='${dbName}'`;
    const dbExistsResult = await adminClient.query(checkDbQuery);

    if (dbExistsResult.rowCount > 0) {
      console.log(`ℹ️  Database "${dbName}" already exists`);
      return false;
    }

    // Create the new database
    await adminClient.query(`CREATE DATABASE "${dbName}"`);
    console.log(`✅ Database "${dbName}" created successfully`);

    // Connect to the newly created database to create the 'tweets' table
    const dbClient = new Client({
      user: process.env.DB_USER,
      host: process.env.DB_HOST,
      database: dbName,
      password: process.env.DB_PASSWORD,
      port: process.env.DB_PORT,
    });

    await dbClient.connect();

    // Create the 'tweets' table
    const createTableQuery = `
      CREATE TABLE IF NOT EXISTS tweets (
        tweet_id VARCHAR(50) PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        tweet TEXT NOT NULL
      );
    `;
    await dbClient.query(createTableQuery);
    console.log(`✅ Table created successfully in "${dbName}"`);

    await dbClient.end();
    return true;
  } catch (error) {
    console.error("❌ Error creating database or table:", error);
    return false;
  } finally {
    await adminClient.end();
  }
}
