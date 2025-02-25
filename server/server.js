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

const TWITTER_USERNAME = process.env.TWITTER_USERNAME;
const TWITTER_PASSWORD = process.env.TWITTER_PASSWORD;

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

  const filePath = join(__dirname, `${data}/${func}.json`);

  if (!fileExistsAndNotEmpty(filePath)) {
    try {
      unlinkSync(filePath); // Delete empty file if exists
    } catch (err) {}
  }

  const command = `python3 ${func}.py ${data}`;
  exec(command, (error, stdout, stderr) => {
    if (error) {
      return res.status(500).json({ error: stderr || error.message });
    }
    res.json({ result: stdout.trim() });
  });
});

// Method: state
app.post("/state", (req, res) => {
  const { function: func, data } = req.body;
  if (!func || !data) {
    return res.status(400).json({ error: "Missing function or data" });
  }

  const filePath = join(__dirname, `${data}/${func}.json`);

  if (!existsSync(filePath)) {
    return res.status(404).json({ error: "File not found" });
  }

  const fileContent = readFileSync(filePath, "utf-8");
  res.json({ state: JSON.parse(fileContent) });
});

// Method: trigger
app.post("/trigger", async (req, res) => {
  const { data } = req.body;
  if (!data) {
    return res.status(400).json({ error: "Missing data" });
  }

  const tweetsPath = join(__dirname, `${data}/tweets.json`);

  // Check if tweets.json exists and is recent (within last 1 hour)
  if (existsSync(tweetsPath)) {
    const lastModified = statSync(tweetsPath).mtime;
    const oneHourAgo = Date.now() - 60 * 60 * 1000;

    if (lastModified.getTime() > oneHourAgo) {
      return res.json({ success: true, message: "Tweets already up to date" });
    }
  }

  const twitterUsername = data; // Assuming 'data' is the Twitter username

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
