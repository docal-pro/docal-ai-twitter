require("dotenv").config();
const { Scraper } = require("agent-twitter-client");

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
  const tweetId = process.argv[2];
  if (!tweetId) {
    console.error("Please provide a tweet ID as an argument");
    process.exit(1);
  }

  try {
    const scraper = new Scraper();
    await scraper.login(
      process.env.TWITTER_USERNAME,
      process.env.TWITTER_PASSWORD
    );

    const tweet = await scraper.getTweet(tweetId);
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
