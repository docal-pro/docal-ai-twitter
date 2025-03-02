import dotenv from "dotenv";
import fs from "fs";
import { Scraper } from "agent-twitter-client";
import { Cookie } from "tough-cookie";

dotenv.config();

const MAX_TWEETS_IN_BATCH = process.env.MAX_TWEETS_IN_BATCH || 10;

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
  const username = process.argv[2];
  console.log("ℹ️  Starting agent...");
  if (!username) {
    console.error("🔎 Please provide a username as an argument");
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
      } else {
        console.log("ℹ️  No cookies found, logging in...");
      }
      // Get forceLoginWithCookies from command line arguments
      const forceLoginWithCookies = process.argv[3];

      if (forceLoginWithCookies === "true") {
        // Load keys from cookies
        let authToken, ct0, guestId;
        if (cookies.length > 0) {
          authToken = cookies.find((c) => c.name === "auth_token");
          ct0 = cookies.find((c) => c.name === "ct0");
          guestId = cookies.find((c) => c.name === "guest_id");
        }
        await scraper.login(
          TWITTER_USERNAME,
          TWITTER_PASSWORD,
          TWITTER_EMAIL,
          TWITTER_2FA_SECRET,
          TWITTER_COOKIES_AUTH_TOKEN || authToken,
          TWITTER_COOKIES_CT0 || ct0,
          TWITTER_COOKIES_GUEST_ID || guestId
        );
      } else if (forceLoginWithCookies === "false") {
        if (cookies.length > 0) await scraper.setCookies(cookies);
        await scraper.login(
          TWITTER_USERNAME,
          TWITTER_PASSWORD,
          TWITTER_EMAIL,
          TWITTER_2FA_SECRET
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
      if (fs.existsSync(cookiesFile)) {
        await scraper.setCookies(loadCookies(cookiesFile));
        console.log("✅ Already logged in");
      } else {
        console.log("❌ Logged in but no cookies found");
      }
    }

    const tweets = scraper.getTweets(username, MAX_TWEETS_IN_BATCH);
    console.log("✅ Tweets fetched for " + username);
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
