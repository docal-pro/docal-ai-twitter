import express, { json } from "express";
import {
  existsSync,
  statSync,
  unlinkSync,
  readFileSync,
  writeFileSync,
} from "fs";
import { exec } from "child_process";
import { join } from "path";
import axios from "axios";
import dotenv from "dotenv";
import { checkDatabase, createDatabase, getAdminClient } from "./database.js";
import { fakeUsers } from "./utils.js";
import { fileURLToPath } from "url";
import { dirname } from "path";
import cors from "cors";

dotenv.config();
const { get } = axios;

// Add these lines near the top of the file after imports
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Twitter/X API Bearer Token
const TWITTER_BEARER_TOKEN = process.env.TWITTER_BEARER_TOKEN;
const DATA_AGE_LIMIT = process.env.DATA_AGE_LIMIT || 24;

// Add CORS configuration
const ALLOWED_ORIGINS = [
  "http://localhost:8080", // Local development
  "https://proxy.docal.com", // Production domain
];

const corsOptions = {
  origin: function (origin, callback) {
    if (!origin || ALLOWED_ORIGINS.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error("Not allowed by CORS"));
    }
  },
  credentials: true,
};

const app = express();
app.use(cors(corsOptions)); // Add CORS middleware
app.use(json());

// Utility function to check if a file exists and is not empty
const fileExistsAndNotEmpty = (filePath) => {
  return existsSync(filePath) && statSync(filePath).size > 0;
};

// Check if database exists and create it if it doesn't
checkDatabase().then(async (exists) => {
  if (!exists) {
    console.log("ℹ️  Creating database...");
    await createDatabase();
  } else {
    console.log("✅ Database already exists");
  }
});

// Method: Get all records from the 'twitter_score' table
app.get("/db", async (req, res) => {
  console.log("🔎 Fetching data from table...");
  try {
    const client = getAdminClient();
    await client.connect();
    const result = await client.query("SELECT * FROM twitter_score");
    await client.end();
    res.json({
      columns: result.fields.map((field) => field.name),
      rows: result.rows.length > 0 ? result.rows : fakeUsers,
    });
  } catch (error) {
    console.error("❌ Error fetching data from 'twitter_score' table:", error);
    res.status(500).json({ error: "An error occurred while fetching data" });
  }
});

// Method: Process an investigate request
app.post("/process", (req, res) => {
  console.log("🔎 Processing request...");
  const { func, user, data, ctxs } = req.body;

  if (!func || !data) {
    console.log("❌ Missing function or data");
    return res.status(400).json({ error: "Missing function or data" });
  }

  let filePath;
  let tweetIds;
  let username;
  if (func !== "scraper" && func !== "indexer") {
    username = data;
    filePath = join(__dirname, `results/${username}/${func}.csv`);
  } else if (func === "indexer") {
    username = data;
    filePath = join(__dirname, `tweets/${username}/tweets.json`);
  } else if (func === "scraper") {
    filePath = join(__dirname, `tweets/${user}/tweets.json`);
    tweetIds = data;
  }

  if (func !== "scraper" && !fileExistsAndNotEmpty(filePath)) {
    try {
      unlinkSync(filePath); // Delete empty file if exists
    } catch (err) {}
  }

  let flag = "false";
  const command =
    func !== "scraper"
      ? func !== "indexer"
        ? func === "classifier"
          ? `python3 src/${func}.py ${username} "${ctxs}"` // Classifier: needs contexts
          : `python3 src/${func}.py ${username}` // Other functions: don't need contexts or flags
        : `python3 src/${func}.py ${username} ${flag} "${ctxs}"` // Indexer: needs flag and contexts
      : `python3 src/${func}.py ${tweetIds} ${user} ${flag} "${ctxs}"`; // Scraper: needs flag and contexts
  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.log(error);
      return res.status(500).json({ error: stderr || error.message });
    }
    res.json({ result: stdout.trim() });
  });
});

// Method: Get the state of an investigation
app.post("/state", (req, res) => {
  console.log("🔎 Getting state...");
  const { user } = req.body;
  if (!user) {
    return res.status(400).json({ error: "Missing user" });
  }

  const resultsPath = join(__dirname, `results/${user}`);
  const tweetsPath = join(__dirname, `tweets/${user}/input.json`);

  const state = {
    results: {},
    tweets: 0,
  };

  // Get counts from CSV files in results directory
  if (existsSync(resultsPath)) {
    const files = readdirSync(resultsPath).filter((file) =>
      file.endsWith(".csv")
    );

    files.forEach((file) => {
      const filePath = join(resultsPath, file);
      const content = readFileSync(filePath, "utf-8");
      const lines = content.split("\n").filter((line) => line.trim());
      state.results[file] = Math.max(0, lines.length - 1); // Subtract header row
    });
  }

  // Get count of items in tweets JSON file
  if (existsSync(tweetsPath)) {
    const tweetsContent = readFileSync(tweetsPath, "utf-8");
    try {
      const tweets = JSON.parse(tweetsContent);
      state.tweets = tweets.length;
    } catch (err) {
      console.error("Error parsing tweets JSON:", err);
      state.tweets = 0;
    }
  }

  res.json({ state });
});

// Method: Trigger data indexing
app.post("/trigger", async (req, res) => {
  console.log("🔎 Triggering data indexing...");
  return res.status(502).json({ error: "Method currently unavailable" });
  const { user } = req.body;
  if (!user) {
    return res.status(400).json({ error: "Missing user" });
  }

  const tweetsPath = join(__dirname, `tweets/${user}/tweets.json`);

  // Check if tweets.json exists and is recent (within last N hours)
  if (existsSync(tweetsPath)) {
    const lastModified = statSync(tweetsPath).mtime;
    const dataAgeLimitAgo = Date.now() - DATA_AGE_LIMIT * 60 * 60 * 1000;

    if (lastModified.getTime() > dataAgeLimitAgo) {
      return res.json({ success: true, message: "Tweets already up to date" });
    }
  }

  const twitterUsername = user; // Assuming 'user' is the Twitter username

  try {
    const response = await get(
      `https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name=${twitterUsername}`,
      {
        headers: {
          Authorization: `Bearer ${TWITTER_BEARER_TOKEN}`,
        },
      }
    );

    writeFileSync(tweetsPath, JSON.stringify(response.data, null, 2));
    res.json({ success: true, message: "Tweets updated" });
  } catch (error) {
    res.status(500).json({ error: error.response?.data || error.message });
  }
});

// Ping endpoint
app.get("/ping", (req, res) => {
  console.log("🔎 Pinging server...");
  res.json({ success: true, message: "✅ Server is running" });
});

const PORT = process.env.NODE_PORT || 3031;
app.listen(PORT, () => console.log(`✅ Server running on port ${PORT}`));
