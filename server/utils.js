// Fake Users
export const fakeUsernames = [
  "@elonmusk",
  "@jack",
];

export const fakeUsers = fakeUsernames.map((username, i) => ({
  id: i + 1,
  username,
  tweet_count: 0,
  score: 0,
  trust: 0,
  investigate: 0,
}));
