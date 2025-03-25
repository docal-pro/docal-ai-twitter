import dotenv from "dotenv";
import fs from "fs";
import { Scraper } from "agent-twitter-client";
import { Cookie } from "tough-cookie";
import path from "path";

dotenv.config();

const TWITTER_USERNAME = process.env.TWITTER_USERNAME;
const TWITTER_PASSWORD = process.env.TWITTER_PASSWORD;
const TWITTER_EMAIL = process.env.TWITTER_EMAIL;
const TWITTER_2FA_SECRET = process.env.TWITTER_2FA_SECRET;
let TWITTER_COOKIES_AUTH_TOKEN = process.env.TWITTER_COOKIES_AUTH_TOKEN;
let TWITTER_COOKIES_CT0 = process.env.TWITTER_COOKIES_CT0;
let TWITTER_COOKIES_GUEST_ID = process.env.TWITTER_COOKIES_GUEST_ID;

const cookiesFile = "logs/cookies.json";

const loadCookies = (cookiesFile) => {
  const load = fs.readFileSync(cookiesFile, "utf-8");
  const parsed = JSON.parse(load);
  return parsed.reduce((acc, current) => {
    const cookie = Cookie.fromJSON(current).cookieString();
    acc.push(cookie);
    return acc;
  }, []);
};

const getCircularReplacer = () => {
  const seen = new WeakSet();
  return (key, value) => {
    if (typeof value === "object" && value !== null) {
      if (seen.has(value)) {
        return "[Circular]";
      }
      seen.add(value);
    }
    return value;
  };
};

async function main() {
  console.log("✅ Scraper environment variables loaded");
  const tweetIds = process.argv[2].split(",");
  console.log("ℹ️  Starting agent...");
  if (!tweetIds) {
    console.error("❌ Please provide list of tweet IDs as an argument");
    process.exit(1);
  }

  try {
    console.log("ℹ️  Initialising scraper...");
    const scraper = new Scraper();
    const isLoggedIn = await scraper.isLoggedIn();

    if (!isLoggedIn) {
      console.log("ℹ️  Trying to log in with cookies...");
      // Check if cookies exist
      let cookies = [];
      if (fs.existsSync(cookiesFile)) {
        cookies = loadCookies(cookiesFile);
      }

      // Check for forceLoginWithCookies flag
      const forceLoginWithCookies = process.argv[3];
      if (forceLoginWithCookies === "strict") {
        console.log("ℹ️  Loading cookies from .env...");
        await scraper.login(
          TWITTER_USERNAME,
          TWITTER_PASSWORD,
          TWITTER_EMAIL,
          TWITTER_2FA_SECRET,
          TWITTER_COOKIES_AUTH_TOKEN,
          TWITTER_COOKIES_CT0,
          TWITTER_COOKIES_GUEST_ID
        );
      } else if (forceLoginWithCookies === "true") {
        if (cookies.length > 0) {
          console.log("ℹ️  Cookies found, logging in with cookies...");
          const authToken = cookies.find((c) => c.name === "auth_token");
          const ct0 = cookies.find((c) => c.name === "ct0");
          const guestId = cookies.find((c) => c.name === "guest_id");
          console.log("⦿ AUTH_TOKEN =", authToken);
          console.log("⦿ CT0 =", ct0);
          console.log("⦿ GUEST_ID =", guestId);
          await scraper.setCookies(cookies);
          await scraper.login(
            TWITTER_USERNAME,
            TWITTER_PASSWORD,
            TWITTER_EMAIL,
            TWITTER_2FA_SECRET
          );
        } else {
          console.log("ℹ️  No cookies found, using values from .env...");
          await scraper.login(
            TWITTER_USERNAME,
            TWITTER_PASSWORD,
            TWITTER_EMAIL,
            TWITTER_2FA_SECRET,
            TWITTER_COOKIES_AUTH_TOKEN,
            TWITTER_COOKIES_CT0,
            TWITTER_COOKIES_GUEST_ID
          );
        }
      } else if (forceLoginWithCookies === "false") {
        if (cookies.length > 0) {
          console.log("ℹ️  Cookies found, logging in with cookies...");
          await scraper.setCookies(cookies);
        } else {
          console.log("ℹ️  No cookies found, attempting fresh login...");
        }
        await scraper.login(
          TWITTER_USERNAME,
          TWITTER_PASSWORD,
          TWITTER_EMAIL,
          TWITTER_2FA_SECRET
        );
      } else {
        console.log("ℹ️  Cookies not requested, attempting fresh login...");
        await scraper.login(
          TWITTER_USERNAME,
          TWITTER_PASSWORD,
          TWITTER_EMAIL,
          TWITTER_2FA_SECRET
        );
      }

      console.log("ℹ️  Backing up cookies...");
      // Ensure the logs directory exists
      const logsDir = path.dirname(cookiesFile);
      if (!fs.existsSync(logsDir)) {
        fs.mkdirSync(logsDir, { recursive: true });
      }
      await scraper.getCookies().then((cookies) => {
        fs.writeFileSync(cookiesFile, JSON.stringify(cookies));
      });
      console.log("✅ Logged in");
    } else {
      if (fs.existsSync(cookiesFile)) {
        await scraper.setCookies(loadCookies(cookiesFile));
        console.log("✅ Already logged in");
      } else {
        console.log("⚠️  Logged in but no cookies found");
      }
    }

    const tweets = [];
    for (const tweetId of tweetIds) {
      const tweet = await scraper.getTweet(tweetId);
      tweets.push(tweet);
    }
    console.log("✅ Tweets fetched");

    // Print the tweet data as JSON string, handling circular references
    console.log(JSON.stringify(tweets, getCircularReplacer()));
    if (isLoggedIn) scraper.logout();
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
