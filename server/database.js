import pkg from "pg";
import dotenv from "dotenv";

const { Client } = pkg;
dotenv.config();

/**
 * Creates a new PostgreSQL client without connecting to a specific database.
 * Used for checking and creating databases.
 */
export function getAdminClient() {
  return new Client({
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: "postgres", // Connect to default postgres database instead of DB_NAME
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

    const response = await client.query(
      "SELECT 1 FROM pg_database WHERE datname = $1",
      [process.env.DB_NAME]
    );

    // Check if tables exist and their names are "twitter" and "twitter_score"
    const tablesResponse = await client.query(
      "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    );

    const tableNames = tablesResponse.rows.map((row) => row.table_name);
    const requiredTables = ["twitter", "twitter_score"];
    const hasAllTables = requiredTables.every((table) =>
      tableNames.includes(table)
    );

    if (!hasAllTables) {
      console.log(
        "ℹ️  Missing required tables ('twitter' and 'twitter_score')"
      );
      return false;
    }

    return response.rowCount > 0;
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
  const client = getAdminClient();
  let dbClient = null;

  try {
    await client.connect();

    // Check if the database already exists
    const dbName = process.env.DB_NAME;
    const checkDbQuery = `SELECT 1 FROM pg_database WHERE datname='${dbName}'`;
    const dbExistsResult = await client.query(checkDbQuery);

    if (dbExistsResult.rowCount > 0) {
      console.log(`ℹ️  Database "${dbName}" already exists`);
    } else {
      // Create the new database
      await client.query(`CREATE DATABASE "${dbName}"`);
      console.log(`✅ Database "${dbName}" created successfully`);
    }

    // Close the admin connection
    await client.end();

    // Create a new connection to the specific database to create tables
    dbClient = new Client({
      host: process.env.DB_HOST,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      database: dbName,
    });
    await dbClient.connect();

    // Create the 'tweets' table
    const createTweetsTableQuery = `
      CREATE TABLE IF NOT EXISTS twitter (
        tweet_id VARCHAR(50) PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        tweet TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
    `;
    await dbClient.query(createTweetsTableQuery);
    console.log(`✅ Table "twitter" created successfully in "${dbName}"`);

    // Create the 'score' table
    const createTwitterScoreTableQuery = `
      CREATE TABLE IF NOT EXISTS twitter_score (
        username VARCHAR(100) NOT NULL,
        tweet_count INT NOT NULL,
        score SMALLINT CHECK (score BETWEEN 0 AND 100),
        trust SMALLINT CHECK (trust BETWEEN 0 AND 5),
        investigate SMALLINT CHECK (investigate BETWEEN 0 AND 4),
        contexts TEXT[] NOT NULL,
        timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
    `;
    await dbClient.query(createTwitterScoreTableQuery);
    console.log(`✅ Table "twitter_score" created successfully in "${dbName}"`);

    return true;
  } catch (error) {
    console.error("❌ Error creating database or table:", error);
    return false;
  } finally {
    if (dbClient) await dbClient.end();
    if (client && client._connected) await client.end();
  }
}
