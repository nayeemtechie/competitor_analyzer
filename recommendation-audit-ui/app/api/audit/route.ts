import { analyzeHtml, buildResult } from '@/lib/audit';
import { applyLlmAnalysis } from '@/lib/llm-analysis';

export const runtime = 'edge';

type AuditRequest = {
  url?: unknown;
  maxPages?: unknown;
  llmProvider?: unknown;
  apiKey?: unknown;
};

const MAX_HTML_BYTES = 2_000_000;
const FETCH_TIMEOUT_MS = 12_000;

function validateTarget(input: unknown): URL {
  if (typeof input !== 'string' || !input.trim()) throw new Error('Enter a valid ecommerce site URL.');
  const url = new URL(input.trim());
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Only public HTTP and HTTPS URLs can be audited.');
  if (url.username || url.password) throw new Error('URLs containing credentials are not supported.');
  if (url.port && !['80', '443'].includes(url.port)) throw new Error('Only standard web ports can be audited.');
  const hostname = url.hostname.replace(/^\[|\]$/g, '').toLowerCase();
  if (hostname === 'localhost' || hostname.endsWith('.localhost') || hostname.endsWith('.local')) {
    throw new Error('Only public ecommerce sites can be audited.');
  }
  if (/^(?:127\.|10\.|0\.|169\.254\.|192\.168\.)/.test(hostname)) throw new Error('Private network addresses are not supported.');
  const private172 = hostname.match(/^172\.(\d{1,3})\./);
  if (private172 && Number(private172[1]) >= 16 && Number(private172[1]) <= 31) throw new Error('Private network addresses are not supported.');
  if (hostname === '::1' || /^f[cd][0-9a-f]:/i.test(hostname) || /^fe8[0-9a-f]:/i.test(hostname)) throw new Error('Private network addresses are not supported.');
  url.hash = '';
  return url;
}

async function fetchPage(input: URL): Promise<{ html: string; url: string }> {
  let current = input;
  for (let redirect = 0; redirect <= 4; redirect += 1) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    try {
      const response = await fetch(current, {
        redirect: 'manual',
        signal: controller.signal,
        headers: {
          Accept: 'text/html,application/xhtml+xml',
          'User-Agent': 'RecommendationAudit/1.0 (+evidence-only storefront audit)',
        },
      });
      if ([301, 302, 303, 307, 308].includes(response.status)) {
        const location = response.headers.get('location');
        if (!location) throw new Error('The site returned an invalid redirect.');
        current = validateTarget(new URL(location, current).toString());
        continue;
      }
      if (!response.ok) throw new Error(`Page returned HTTP ${response.status}.`);
      const contentType = response.headers.get('content-type') ?? '';
      if (!/text\/html|application\/xhtml\+xml/i.test(contentType)) throw new Error('Page did not return HTML.');
      const declaredLength = Number(response.headers.get('content-length') ?? 0);
      if (declaredLength > MAX_HTML_BYTES) throw new Error('Page HTML is too large to inspect safely.');
      const html = await response.text();
      if (new TextEncoder().encode(html).byteLength > MAX_HTML_BYTES) throw new Error('Page HTML is too large to inspect safely.');
      return { html, url: current.toString() };
    } finally {
      clearTimeout(timer);
    }
  }
  throw new Error('The site redirected too many times.');
}

function priority(url: string): number {
  const path = new URL(url).pathname.toLowerCase();
  if (/\/(?:products?|p)\//.test(path)) return 0;
  if (/\/(?:collections?|categories?|catalog|shop)(?:\/|$)/.test(path)) return 1;
  if (/\/(?:cart|search)(?:\/|$)/.test(path)) return 2;
  return 3;
}

function usefulLink(url: string): boolean {
  const parsed = new URL(url);
  return !/(?:\/account|\/login|\/register|\/checkout|\/privacy|\/terms|\/contact|\/blog)(?:\/|$)/i.test(parsed.pathname);
}

export async function POST(request: Request) {
  let body: AuditRequest;
  try {
    body = await request.json() as AuditRequest;
  } catch {
    return Response.json({ error: 'The request body must be valid JSON.' }, { status: 400 });
  }

  let startUrl: URL;
  try {
    startUrl = validateTarget(body.url);
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : 'Invalid URL.' }, { status: 400 });
  }
  const maxPages = Math.min(20, Math.max(1, Number.isFinite(Number(body.maxPages)) ? Math.floor(Number(body.maxPages)) : 8));
  const provider = body.llmProvider === 'openai' ? 'openai' : null;
  const apiKey = provider && typeof body.apiKey === 'string' ? body.apiKey.trim() : '';
  if (provider && !apiKey) return Response.json({ error: 'Enter an OpenAI API key for LLM analysis.' }, { status: 400 });

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      const send = (payload: unknown) => controller.enqueue(encoder.encode(`${JSON.stringify(payload)}\n`));
      void (async () => {
        try {
          const queue = [startUrl.toString()];
          const queued = new Set(queue);
          const visited = new Set<string>();
          const pages: { url: string; pageType: string; placements: ReturnType<typeof analyzeHtml>['placements'] }[] = [];
          const failures: string[] = [];

          send({ type: 'progress', percent: 8, message: 'Discovering storefront pages', currentUrl: startUrl.toString() });
          while (queue.length && pages.length < maxPages) {
            queue.sort((left, right) => priority(left) - priority(right));
            const next = queue.shift();
            if (!next || visited.has(next)) continue;
            visited.add(next);
            const percent = 12 + Math.round((pages.length / maxPages) * 60);
            send({ type: 'progress', percent, message: `Inspecting page ${pages.length + 1} of up to ${maxPages}`, currentUrl: next });
            try {
              const fetched = await fetchPage(validateTarget(next));
              const analysis = analyzeHtml(fetched.html, fetched.url);
              pages.push({ url: fetched.url, pageType: analysis.pageType, placements: analysis.placements });
              for (const link of analysis.links.filter(usefulLink).slice(0, 160)) {
                if (!visited.has(link) && !queued.has(link)) {
                  queued.add(link);
                  queue.push(link);
                }
              }
            } catch (error) {
              failures.push(error instanceof Error ? error.message : 'Page request failed.');
            }
          }

          if (!pages.length) {
            throw new Error(failures[0] || 'No public HTML pages could be analyzed. The site may block automated access.');
          }

          send({ type: 'progress', percent: 78, message: 'Scoring recommendation coverage' });
          let result = buildResult(startUrl.toString(), pages);
          if (failures.length) result.warning = `${failures.length} linked page${failures.length === 1 ? '' : 's'} could not be read; results reflect the pages successfully collected.`;

          if (provider) {
            send({ type: 'progress', percent: 88, message: 'Interpreting collected evidence with OpenAI' });
            try {
              result = await applyLlmAnalysis(result, provider, apiKey);
            } catch {
              result.warning = `${result.warning ? `${result.warning} ` : ''}OpenAI interpretation was unavailable, so the deterministic audit is shown instead.`;
            }
          }

          send({ type: 'progress', percent: 97, message: 'Preparing presales findings' });
          send({ type: 'result', result });
        } catch (error) {
          send({ type: 'error', message: error instanceof Error ? error.message : 'The audit could not be completed.' });
        } finally {
          controller.close();
        }
      })();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    },
  });
}
