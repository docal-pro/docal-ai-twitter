// Default usernames
export const defaultUsernames = [
  "@",
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
    username: "@",
    transaction: "0x0000000000000000000000000000000000000000",
    contexts: [],
    tweet_ids: [],
    timestamp: "1970-01-01 00:00:00.000000+00:00",
  },
];