// Fake usernames
export const fakeUsernames = [
  "@elonmusk",
  "@jack",
];

// Generate fake users from usernames
export const fakeUsers = fakeUsernames.map((username, i) => ({
  id: i + 1,
  username,
  tweet_count: 0,
  score: 0,
  trust: 0,
  investigate: 0,
  contexts: [],
  timestamp: null,
}));

// Fake placeholder schedule
export const fakeSchedule = [
  {
    transaction: "0x0000000000000000000000000000000000000000",
    tweet_ids: ["12345678909876543210", "12345678909876543211"],
    timestamp: null,
  },
];