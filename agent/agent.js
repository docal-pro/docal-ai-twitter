import dotenv from "dotenv";
import { Scraper } from "agent-twitter-client";

dotenv.config();

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
    console.log("ℹ️  Logging in...");
    await scraper.login(
      process.env.TWITTER_USERNAME,
      process.env.TWITTER_PASSWORD,
      process.env.TWITTER_EMAIL,
      process.env.TWITTER_2FA_SECRET,
      process.env.TWITTER_COOKIES_AUTH_TOKEN,
      process.env.TWITTER_COOKIES_CT0,
      process.env.TWITTER_COOKIES_GUEST_ID
    );
    console.log("✅ Logged in");
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
