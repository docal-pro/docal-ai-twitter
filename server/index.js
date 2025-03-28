import express, { json } from "express";
import fs, {
  existsSync,
  statSync,
  unlinkSync,
  readFileSync,
  writeFileSync,
} from "fs";
import { exec } from "child_process";
import { join, dirname } from "path";
import axios from "axios";
import dotenv from "dotenv";
import { checkDatabase, createDatabase, getAdminClient } from "./database.js";
import { defaultUsers, defaultSchedule } from "./utils.js";
import { fileURLToPath } from "url";
import cors from "cors";
import https from "https";

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
    console.log("🔎 CORS origin:", origin || "Same-Origin-Request");
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

// Use existing certificates
const httpsOptions = {
  key: fs.readFileSync("/etc/letsencrypt/live/store.docal.pro/privkey.pem"),
  cert: fs.readFileSync("/etc/letsencrypt/live/store.docal.pro/cert.pem"),
  ca: fs.readFileSync("/etc/letsencrypt/live/store.docal.pro/chain.pem"),
  fullChain: fs.readFileSync(
    "/etc/letsencrypt/live/store.docal.pro/fullchain.pem"
  ),
};

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
    console.log("✅ Database and tables already exist");
  }
});

// Ping endpoint
app.get("/ping", (req, res) => {
  console.log("🔎 Pinging server...");
  res.json({ success: true, message: "✅ Server is running" });
});

// Method: Get all records from the 'twitter_score' table
app.get("/db", async (req, res) => {
  console.log("🔎 Fetching data from score table");
  try {
    const client = getAdminClient();
    await client.connect();
    const result = await client.query("SELECT * FROM twitter_score");
    await client.end();
    res.json({
      columns: result.fields.map((field) => field.name),
      rows: result.rows.length > 0 ? result.rows : defaultUsers,
    });
  } catch (error) {
    console.error("❌ Error fetching data from 'twitter_score' table:", error);
    res.status(500).json({ error: "An error occurred while fetching data" });
  }
});

// Method: Get schedule for a user from the 'schedule' table
app.post("/schedule", async (req, res) => {
  console.log("🔎 Fetching schedule from table for request:\n", req.body);
  const { query } = req.body;
  if (!query) {
    return res.status(400).json({ error: "Missing query" });
  }
  try {
    const client = getAdminClient();
    await client.connect();
    const result = await client.query(
      "SELECT * FROM schedule WHERE caller = $1",
      [query]
    );
    await client.end();
    res.json({
      columns: result.fields.map((field) => field.name),
      rows: result.rows.length > 0 ? result.rows : defaultSchedule,
    });
  } catch (error) {
    console.error("❌ Error fetching data from 'schedule' table:", error);
    res.status(500).json({ error: "An error occurred while fetching schedule" });
  }
});

// Method: Get the state of an investigation
app.post("/state", (req, res) => {
  console.log("🔎 Getting state for request:\n", req.body);
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

// Method: Process an investigate request
app.post("/process", (req, res) => {
  console.log("🔎 Processing request:\n", req.body);
  const { func, user, data, ctxs, caller, transaction } = req.body;

  if (!func) {
    console.log("❌ Missing function");
    return res.status(400).json({ error: "Missing function" });
  }

  let filePath;
  let tweetIds;
  let username;
  if (func !== "scraper" && func !== "indexer") {
    username = user;
    filePath = join(__dirname, `results/${username}/${func}.csv`);
  } else if (func === "indexer") {
    username = user;
    filePath = join(__dirname, `tweets/${username}/input.json`);
  } else if (func === "scraper") {
    username = user;
    filePath = join(__dirname, `tweets/${user}/input.json`);
    tweetIds = data;
  }

  if (func !== "scraper" && !fileExistsAndNotEmpty(filePath)) {
    try {
      unlinkSync(filePath); // Delete empty file if exists
    } catch (err) { }
  }

  let flag = "false";
  let source = "/root/docal-ai-twitter";
  const command =
    func !== "scraper"
      ? func !== "indexer"
        ? func === "classifier"
          ? `python3 ${source}/src/${func}.py ${username} "${ctxs}"` // Classifier: needs contexts
          : `python3 ${source}/src/${func}.py ${username}` // Other functions: don't need contexts or flags
        : `python3 ${source}/src/${func}.py ${username} ${flag} "${ctxs}" ${caller} ${transaction}` // Indexer: needs flag and contexts
      : `python3 ${source}/src/${func}.py ${username} ${tweetIds} ${flag} "${ctxs}" ${caller} ${transaction}`; // Scraper: needs flag and contexts
  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.log(error);
      return res.status(500).json({ error: stderr || error.message });
    }
    res.json({ result: stdout.trim() });
  });
});

// Method: Trigger data indexing
app.post("/trigger", async (req, res) => {
  console.log("🔎 Triggering data indexing for request:\n", req.body);
  return res.status(502).json({ error: "Method currently unavailable" });
  const { user, caller, transaction } = req.body;
  if (!user) {
    return res.status(400).json({ error: "Missing user" });
  }

  const tweetsPath = join(__dirname, `tweets/${user}/input.json`);

  // Check if input.json exists and is recent (within last N hours)
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

const PORT = process.env.NODE_PORT || 3035;
const server = https.createServer(httpsOptions, app);

server.listen(PORT, () => {
  console.log(`✅ HTTPS Server running on port ${PORT}`);
});

// Optional: Redirect HTTP to HTTPS
const httpApp = express();
httpApp.use((req, res) => {
  res.redirect(`https://${req.headers.host.split(":")[0]}:${PORT}${req.url}`);
});

const HTTP_PORT = process.env.HTTP_PORT || 3135;
httpApp.listen(HTTP_PORT, () => {
  console.log(`✅ HTTP Redirect Server running on port ${HTTP_PORT}`);
});
