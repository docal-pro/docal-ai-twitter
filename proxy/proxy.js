export default {
  async fetch(request) {
    try {
      const url = new URL(request.url);
      const path = url.pathname.split("/")[1]; // Extract method name
      const serverUrl = env.SERVER_URL + path; // Server URL with method

      let requestBody = {};
      if (request.method === "POST") {
        requestBody = await request.json();
      }

      const response = await fetch(serverUrl, {
        method: request.method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      return new Response(await response.text(), {
        status: response.status,
        headers: { "Content-Type": "application/json" },
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: err.message }), {
        status: 500,
      });
    }
  },
};
