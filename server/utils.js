// Default usernames
export const defaultUsernames = [
  "@elonmusk",
  "@jack",
];

// Generate default users from usernames
export const defaultUsers = defaultUsernames.map((username, i) => ({
  id: i + 1,
  username,
  tweet_count: 0,
  score: 0,
  trust: 0,
  investigate: 0,
  contexts: [],
  timestamp: null,
}));

// Default placeholder schedule
export const defaultSchedule = [
  {
    caller: "0x0000000000000000000000000000000000000000",
    username: "@elonmusk",
    transaction: "0x0000000000000000000000000000000000000000",
    contexts: ["context1", "context2"],
    tweet_ids: ["12345678909876543210", "12345678909876543211", "12345678909876543212", "12345678909876543213"],
    timestamp: "2025-03-27 00:00:00.000000+00:00",
  },
];