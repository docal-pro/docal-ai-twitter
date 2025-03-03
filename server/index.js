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

dotenv.config();
const { get } = axios;
const app = express();
app.use(json());

// Twitter/X API Bearer Token
const TWITTER_BEARER_TOKEN = process.env.TWITTER_BEARER_TOKEN;
const DATA_AGE_LIMIT = process.env.DATA_AGE_LIMIT || 24;

// Utility function to check if a file exists and is not empty
const fileExistsAndNotEmpty = (filePath) => {
  return existsSync(filePath) && statSync(filePath).size > 0;
};

// Check if database exists and create it if it doesn't
checkDatabase().then((exists) => {
  if (!exists) {
    createDatabase();
    console.log("✅ Database created successfully");
  } else {
    console.log("✅ Database already exists");
  }
});

// Method: Get all records from the 'score' table
app.get("/db", async (req, res) => {
  try {
    const client = getAdminClient();
    await client.connect();
    const result = await client.query("SELECT * FROM score");
    await client.end();
    res.json({
      columns: result.fields.map((field) => field.name),
      rows: result.rows.length > 0 ? result.rows : fakeUsers,
    });
  } catch (error) {
    console.error('❌ Error fetching data from "score" table:', error);
    res.status(500).json({ error: "An error occurred while fetching data" });
  }
});

// Method: Process an investigate request
app.post("/process", (req, res) => {
  const { func, user, data } = req.body;
  if (func === "indexer") {
    return res.status(501).json({ error: "Method currently unavailable" });
  }
  if (!func || !data) {
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
        ? `python3 ${func}.py ${username}` // Flag not needed
        : `python3 ${func}.py ${username} ${flag}` // Indexer needs flag
      : `python3 ${func}.py ${tweetIds} ${user} ${flag}`; // Scraper needs flag
  exec(command, (error, stdout, stderr) => {
    if (error) {
      return res.status(500).json({ error: stderr || error.message });
    }
    res.json({ result: stdout.trim() });
  });
});

// Method: Get the state of an investigation
app.post("/state", (req, res) => {
  const { function: func, user } = req.body;
  if (!func || !user) {
    return res.status(400).json({ error: "Missing function or user" });
  }

  const filePath = join(__dirname, `results/${user}/${func}.csv`);

  if (!existsSync(filePath)) {
    return res.status(404).json({ error: "File not found" });
  }

  const fileContent = readFileSync(filePath, "utf-8");
  res.json({ state: JSON.parse(fileContent) });
});

// Method: Trigger data indexing
app.post("/trigger", async (req, res) => {
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

app.listen(3030, () => console.log("✅ Server running on port 3030"));
