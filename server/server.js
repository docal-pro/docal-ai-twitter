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

dotenv.config();
const { get } = axios;
const app = express();
app.use(json());

// Twitter/X API Bearer Token
const TWITTER_BEARER_TOKEN = process.env.TWITTER_BEARER_TOKEN;

// Utility function to check if a file exists and is not empty
const fileExistsAndNotEmpty = (filePath) => {
  return existsSync(filePath) && statSync(filePath).size > 0;
};

// Method: process
app.post("/process", (req, res) => {
  const { function: func, data } = req.body;
  if (!func || !data) {
    return res.status(400).json({ error: "Missing function or data" });
  }

  let filePath;
  let tweetId;
  let user;
  if (func !== "scraper" && func !== "indexer") {
    filePath = join(__dirname, `results/${user}/${func}.csv`);
    user = data;
  } else if (func === "indexer") {
    filePath = join(__dirname, `tweets/${user}/tweets.json`);
    user = data;
  } else if (func === "scraper") {
    filePath = join(__dirname, `tweets/${user}/input.json`);
    tweetId = data;
  }

  if (!fileExistsAndNotEmpty(filePath)) {
    try {
      unlinkSync(filePath); // Delete empty file if exists
    } catch (err) {}
  }

  const command =
    func !== "scraper"
      ? `python3 ${func}.py ${user}`
      : `python3 ${func}.py ${tweetId}`;
  exec(command, (error, stdout, stderr) => {
    if (error) {
      return res.status(500).json({ error: stderr || error.message });
    }
    res.json({ result: stdout.trim() });
  });
});

// Method: state
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

// Method: trigger
app.post("/trigger", async (req, res) => {
  const { user } = req.body;
  if (!user) {
    return res.status(400).json({ error: "Missing user" });
  }

  const tweetsPath = join(__dirname, `tweets/${user}/tweets.json`);

  // Check if tweets.json exists and is recent (within last 1 hour)
  if (existsSync(tweetsPath)) {
    const lastModified = statSync(tweetsPath).mtime;
    const oneHourAgo = Date.now() - 60 * 60 * 1000;

    if (lastModified.getTime() > oneHourAgo) {
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

app.listen(3030, () => console.log("Server running on port 3030"));
