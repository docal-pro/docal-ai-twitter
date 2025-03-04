export default {
  async fetch(request, env) {
    // Define the list of allowed origins
    const allowedOrigins = ["https://docal.ai", "http://localhost:3000"];

    // Function to check if the origin is allowed
    function isOriginAllowed(origin) {
      return allowedOrigins.includes(origin);
    }

    // Function to handle CORS headers
    function getCorsHeaders(request) {
      const origin = request.headers.get("Origin");
      const headers = {
        "Access-Control-Allow-Methods": "GET,HEAD,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "86400",
      };
      if (isOriginAllowed(origin)) {
        headers["Access-Control-Allow-Origin"] = origin;
      }
      return headers;
    }

    // Handle preflight requests
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: getCorsHeaders(request),
      });
    }

    try {
      const url = new URL(request.url);
      const type = url.pathname.split("/")[1]; // Extract type name
      const path = url.pathname.split("/")[2]; // Extract method name
      const serverUrl =
        env.SERVER_URL +
        ":" +
        (type === "twitter"
          ? env.SERVER_PORT_TWITTER
          : env.SERVER_PORT_DISCOURSE) +
        "/" +
        path; // Server URL with method

      let requestBody = {};
      const headers = { "Content-Type": "application/json" };

      // Only include body for POST requests
      const fetchOptions = {
        method: request.method,
        headers: headers,
      };

      if (request.method === "POST") {
        requestBody = await request.json();
        fetchOptions.body = JSON.stringify(requestBody);
      }

      const response = await fetch(serverUrl, fetchOptions);
      const responseText = await response.text();

      // Forward the actual server status code and response
      const corsHeaders = getCorsHeaders(request);
      const modifiedHeaders = new Headers(response.headers);
      Object.entries(corsHeaders).forEach(([key, value]) => {
        modifiedHeaders.set(key, value);
      });

      return new Response(responseText, {
        status: response.status, // Forward the actual status code
        headers: modifiedHeaders,
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: err.message }), {
        status: 500,
        headers: getCorsHeaders(request),
      });
    }
  },
};
