export default {
  async fetch(request) {
    try {
      const { function: func, data } = await request.json();
      const serverUrl = env.SERVER_URL; // Server URL

      const response = await fetch(serverUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ function: func, data })
      });

      return new Response(await response.text(), {
        status: response.status,
        headers: { "Content-Type": "application/json" }
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: err.message }), { status: 500 });
    }
  }
};

