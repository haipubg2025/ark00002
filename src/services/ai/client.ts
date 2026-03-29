
import { GoogleGenAI } from "@google/genai";
import { AppSettings } from "../../types";

// SAFELY override fetch using defineProperty to handle "only a getter" environments
const originalFetch = window.fetch;
try {
  Object.defineProperty(window, 'fetch', {
    configurable: true,
    enumerable: true,
    get: () => async (...args: [RequestInfo | URL, RequestInit?]) => {
      const [resource, config] = args;
      const url = typeof resource === 'string' ? resource : resource instanceof URL ? resource.href : (resource as Request).url;
      
      const RETRYABLE_STATUS_CODES = [401, 429, 502, 503, 504];
      const MAX_RETRIES = 3;
      
      const performFetch = async (targetUrl: string, targetConfig: RequestInit | undefined, attempt: number = 0): Promise<Response> => {
        try {
          const response = await originalFetch(targetUrl, targetConfig);
          
          if (RETRYABLE_STATUS_CODES.includes(response.status) && attempt < MAX_RETRIES) {
            const delay = response.status === 401 ? 2000 : Math.pow(2, attempt + 1) * 1000;
            // console.log(`%c[Fetch Guard] 🔄 Thử lại lần ${attempt + 1}/${MAX_RETRIES} (Status: ${response.status}) sau ${delay}ms...`, "color: #f59e0b; font-weight: bold;");
            await new Promise(resolve => setTimeout(resolve, delay));
            return performFetch(targetUrl, targetConfig, attempt + 1);
          }
          
          return response;
        } catch (error) {
          if (attempt < MAX_RETRIES) {
            const delay = Math.pow(2, attempt + 1) * 1000;
            // console.log(`%c[Fetch Guard] 🌐 Lỗi mạng, thử lại lần ${attempt + 1}/${MAX_RETRIES} sau ${delay}ms...`, "color: #ef4444; font-weight: bold;");
            await new Promise(resolve => setTimeout(resolve, delay));
            return performFetch(targetUrl, targetConfig, attempt + 1);
          }
          throw error;
        }
      };

      // Check if this is a Google AI request
      if (url.includes('generativelanguage.googleapis.com')) {
        const proxyUrl = (window as Window & { __GEMINI_PROXY_URL__?: string | null }).__GEMINI_PROXY_URL__;
        
        if (proxyUrl) {
          // Strip /v1 or /v1beta from the end of proxyUrl if present
          const cleanProxy = proxyUrl.trim().replace(/\/+$/, '').replace(/\/v1beta$|\/v1$/, '');
          const newUrl = url.replace('https://generativelanguage.googleapis.com', cleanProxy);
          
          // console.log(`%c[Fetch Guard] 🛡️ Redirecting to Proxy: ${newUrl.substring(0, 80)}...`, "color: #f59e0b; font-weight: bold;");
          
          // Ensure headers are set correctly for the proxy
          const newConfig = { ...config };
          if (newConfig.headers) {
            const headers = new Headers(newConfig.headers);
            // Some proxies need the key in Authorization header
            const apiKey = headers.get('x-goog-api-key');
            if (apiKey && !headers.has('Authorization')) {
              headers.set('Authorization', `Bearer ${apiKey}`);
            }
            newConfig.headers = headers;
          }
          
          return performFetch(newUrl, newConfig);
        }
      }

      return performFetch(url, config);
    }
  });
} catch (e) {
  console.error("[AI Client] Không thể ghi đè fetch toàn cục, đang sử dụng fallback SDK.", e);
}

// Global counter for sequential API key rotation
let currentKeyIndex = 0;

// Helper to get configured AI instance
export const getAiClient = (settings?: AppSettings, forceDirect: boolean = false) => {
  const useProxy = settings?.useProxy !== false && settings?.proxyUrl && settings.proxyKey && !forceDirect;

  // Set global proxy URL for the fetch override
  if (useProxy) {
    (window as Window & { __GEMINI_PROXY_URL__?: string | null }).__GEMINI_PROXY_URL__ = settings.proxyUrl;
  } else {
    (window as Window & { __GEMINI_PROXY_URL__?: string | null }).__GEMINI_PROXY_URL__ = null;
  }

  // Priority: Proxy Key > Personal API Key > System API Key
  // If forceDirect is true, we skip Proxy entirely
  let apiKey: string = "";
  let source = "SYSTEM";
  
  if (useProxy) {
    apiKey = settings.proxyKey!;
    source = "PROXY";
  } else if (settings?.useGeminiApi !== false && settings?.geminiApiKey && Array.isArray(settings.geminiApiKey)) {
    const keys = settings.geminiApiKey.filter(k => k && k.trim() !== "" && k !== "YOUR_API_KEY");
    if (keys.length > 0) {
        // Sequential rotation (Round Robin)
        const index = currentKeyIndex % keys.length;
        apiKey = keys[index];
        source = `PERSONAL_LIST (Key #${index + 1})`;
        currentKeyIndex = (index + 1) % keys.length;
    }
  }

  // Fallback to system key if no valid key found in settings
  if (!apiKey) {
    apiKey = process.env.GEMINI_API_KEY || "";
    source = "SYSTEM_ENV";
  }
  
  const requestOptions: { headers?: Record<string, string> } = {};
  let baseUrl: string | undefined = undefined;
  
  if (useProxy) {
    // Sanitize proxy URL (remove trailing slash and common version suffixes)
    baseUrl = settings.proxyUrl!.trim();
    if (baseUrl.endsWith('/')) {
      baseUrl = baseUrl.slice(0, -1);
    }
    
    // SDK appends version (e.g. /v1beta) automatically. 
    // If user provided it, remove it to avoid double versioning (e.g. /v1beta/v1beta)
    if (baseUrl.endsWith('/v1beta')) {
      baseUrl = baseUrl.slice(0, -7);
    } else if (baseUrl.endsWith('/v1')) {
      baseUrl = baseUrl.slice(0, -3);
    }
    
    // Some proxies require Authorization header instead of x-goog-api-key
    requestOptions.headers = {
      'Authorization': `Bearer ${apiKey}`,
      // Keep x-goog-api-key for standard Gemini proxies
      'x-goog-api-key': apiKey
    };
  }

  // CRITICAL: baseUrl/baseURL must be at the top level of the config object for @google/genai SDK
  // We provide both to be safe across different SDK versions
  const genAIConfig: Record<string, unknown> = {
    apiKey: apiKey,
  };

  if (!apiKey && !baseUrl) {
    console.error(`%c[AI Client] ❌ KHÔNG TÌM THẤY API KEY! (Source: ${source})`, "color: #ef4444; font-weight: bold;");
  } else {
    // const maskedKey = apiKey ? `${apiKey.substring(0, 4)}...${apiKey.substring(apiKey.length - 4)}` : "NONE";
    // console.log(`%c[AI Client] 🔑 Sử dụng key từ: ${source} (${maskedKey})`, "color: #10b981; font-weight: bold;");
  }

  if (baseUrl) {
    // console.log(`%c[AI Client] 🌐 Đang cấu hình PROXY: ${baseUrl}`, "color: #38bdf8; font-weight: bold;");
    
    // Set at top level - @google/genai uses baseURL (uppercase URL)
    genAIConfig.baseURL = baseUrl;
    genAIConfig.baseUrl = baseUrl;
    genAIConfig.apiEndpoint = baseUrl;
    
    // Set inside requestOptions as well for redundancy
    genAIConfig.requestOptions = {
      ...requestOptions,
      baseURL: baseUrl,
      baseUrl: baseUrl,
      apiEndpoint: baseUrl,
      customHeaders: requestOptions.headers,
      fetch: (url: string, options: RequestInit) => {
        // If it's a relative URL or already contains the proxy, let it be
        if (url.includes(baseUrl!)) return fetch(url, options);
        
        // If it's a Google API URL, replace it
        const googleBase = 'https://generativelanguage.googleapis.com';
        if (url.startsWith(googleBase)) {
          const proxyUrl = url.replace(googleBase, baseUrl!);
          // console.log(`%c[AI Client] 🚀 Proxy Redirect: ${proxyUrl.substring(0, 80)}...`, "color: #a855f7; font-size: 10px;");
          return fetch(proxyUrl, options);
        }
        
        // Fallback for any other absolute URL that might be internal to the SDK
        try {
          const urlObj = new URL(url);
          if (urlObj.hostname === 'generativelanguage.googleapis.com') {
            const proxyUrl = url.replace(urlObj.origin, baseUrl!);
            return fetch(proxyUrl, options);
          }
        } catch (e) {
          console.error("Proxy URL parsing error:", e);
        }

        return fetch(url, options);
      }
    };

    // Task: Force fetch to use the proxy URL by overriding the global fetch if necessary
    // but for now we rely on the SDK's apiEndpoint property which is most reliable
  } else {
    // console.log("%c[AI Client] 🔑 Đang sử dụng API KEY TRỰC TIẾP", "color: #10b981; font-weight: bold;");
    if (Object.keys(requestOptions).length > 0) {
      genAIConfig.requestOptions = requestOptions;
    }
  }

  return new GoogleGenAI(genAIConfig);
};

// Default instance for backward compatibility (uses env key)
const defaultKey = process.env.GEMINI_API_KEY || "no-key";
export const ai = new GoogleGenAI({ apiKey: defaultKey });
