import dotenv from "dotenv";
import fs from "fs";
import { Scraper } from "agent-twitter-client";
import { Cookie } from "tough-cookie";

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
  const tweetId = process.argv[2];
  console.log("ℹ️  Starting agent...");
  if (!tweetId) {
    console.error("🔎 Please provide a tweet ID as an argument");
    process.exit(1);
  }

  try {
    console.log("ℹ️  Initialising scraper...");
    const scraper = new Scraper();
    const isLoggedIn = await scraper.isLoggedIn();

    if (!isLoggedIn) {
      console.log("ℹ️  Trying to log in with cookies...");
      // Check if cookies exist
      if (fs.existsSync(cookiesFile)) {
        await scraper.setCookies(loadCookies(cookiesFile));
      } else {
        console.log("ℹ️  No cookies found, logging in...");
      }
      // Check if forceLoginWithCookies is true
      const forceLoginWithCookies = process.argv[3] === "true";

      if (forceLoginWithCookies) {
        await scraper.login(
          TWITTER_USERNAME,
          TWITTER_PASSWORD,
          TWITTER_EMAIL,
          TWITTER_2FA_SECRET,
          TWITTER_COOKIES_AUTH_TOKEN,
          TWITTER_COOKIES_CT0,
          TWITTER_COOKIES_GUEST_ID
        );
      } else {
        await scraper.login(
          TWITTER_USERNAME,
          TWITTER_PASSWORD,
          TWITTER_EMAIL,
          TWITTER_2FA_SECRET
        );
      }

      await scraper.getCookies().then((cookies) => {
        fs.writeFileSync(cookiesFile, JSON.stringify(cookies));
      });
      console.log("✅ Logged in");
    } else {
      await scraper.setCookies(loadCookies(cookiesFile));
      console.log("✅ Already logged in");
    }

    const tweet = await scraper.getTweet(tweetId);
    console.log("✅ Tweet fetched");
    // Print the tweet data as JSON string, handling circular references
    console.log(JSON.stringify(tweet, getCircularReplacer()));
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
