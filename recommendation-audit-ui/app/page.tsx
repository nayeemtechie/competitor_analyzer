'use client';

import {
  AlertCircle,
  ArrowUpRight,
  CheckCircle2,
  Eye,
  EyeOff,
  FileSearch,
  Globe2,
  Lightbulb,
  LoaderCircle,
  LockKeyhole,
  Radar,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Target,
} from 'lucide-react';
import { useEffect, useMemo, useState, type ReactNode, type SubmitEvent } from 'react';

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { NativeSelect, NativeSelectOption } from '@/components/ui/native-select';
import { Progress } from '@/components/ui/progress';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

type AuditStatus = 'idle' | 'running' | 'done' | 'error';

type Placement = {
  pageUrl: string;
  pageType: string;
  title: string;
  placementPosition: string;
  detectedType: string;
  productCount: number;
  confidence: number;
  evidence: string[];
};

type Finding = {
  title: string;
  detail: string;
  evidenceUrls: string[];
};

type Opportunity = Finding & { priority: 'High' | 'Medium' | 'Low' };

type AuditResult = {
  siteUrl: string;
  completedAt: string;
  overallScore: number;
  pagesAnalyzed: { url: string; pageType: string }[];
  placements: Placement[];
  coverage: {
    pageType: string;
    pagesWithPlacements: number;
    totalPages: number;
    percentage: number;
  }[];
  strengths: Finding[];
  gaps: Finding[];
  opportunities: Opportunity[];
  evidenceUrls: string[];
  summary?: string;
  llmUsed: boolean;
  warning?: string;
};

const initialProgress = {
  percent: 0,
  message: 'Preparing audit',
  currentUrl: '',
};

function SectionTitle({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
      <span className="text-emerald-600">{icon}</span>
      {children}
    </div>
  );
}

function PriorityBadge({ priority }: { priority: Opportunity['priority'] }) {
  const styles = {
    High: 'border-rose-200 bg-rose-50 text-rose-700',
    Medium: 'border-amber-200 bg-amber-50 text-amber-700',
    Low: 'border-slate-200 bg-slate-50 text-slate-600',
  };
  return <Badge className={styles[priority]} variant="outline">{priority}</Badge>;
}

function EvidenceLinks({ urls }: { urls: string[] }) {
  if (!urls.length) return null;
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {urls.slice(0, 3).map((url) => (
        <a
          className="inline-flex max-w-full items-center gap-1 rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-200"
          href={url}
          key={url}
          rel="noreferrer"
          target="_blank"
        >
          <span className="max-w-52 truncate">{new URL(url).pathname || '/'}</span>
          <ArrowUpRight className="size-3" />
        </a>
      ))}
    </div>
  );
}

export default function Home() {
  const [siteUrl, setSiteUrl] = useState('');
  const [maxPages, setMaxPages] = useState(8);
  const [provider, setProvider] = useState('none');
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [status, setStatus] = useState<AuditStatus>('idle');
  const [progress, setProgress] = useState(initialProgress);
  const [result, setResult] = useState<AuditResult | null>(null);
  const [error, setError] = useState('');

  const coveredPageTypes = useMemo(
    () => result?.coverage.filter((item) => item.pagesWithPlacements > 0).length ?? 0,
    [result],
  );

  async function executeAudit(auditUrl: string, auditMaxPages: number, auditProvider: string, auditApiKey: string) {
    setSiteUrl(auditUrl);
    setMaxPages(auditMaxPages);
    setProvider(auditProvider);
    setStatus('running');
    setResult(null);
    setError('');
    setProgress({ percent: 4, message: 'Starting site discovery', currentUrl: auditUrl });

    try {
      const response = await fetch('/api/audit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: auditUrl,
          maxPages: auditMaxPages,
          llmProvider: auditProvider === 'openai' ? 'openai' : null,
          apiKey: auditProvider === 'openai' ? auditApiKey : null,
        }),
      });

      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => null) as { error?: string } | null;
        throw new Error(payload?.error || 'The audit could not be started.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffered = '';

      while (true) {
        const { value, done } = await reader.read();
        buffered += decoder.decode(value ?? new Uint8Array(), { stream: !done });
        const lines = buffered.split('\n');
        buffered = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.trim()) continue;
          const update = JSON.parse(line);
          if (update.type === 'progress') {
            setProgress({
              percent: update.percent,
              message: update.message,
              currentUrl: update.currentUrl ?? '',
            });
          } else if (update.type === 'result') {
            setResult(update.result);
            setProgress({ percent: 100, message: 'Audit complete', currentUrl: '' });
            setStatus('done');
          } else if (update.type === 'error') {
            throw new Error(update.message);
          }
        }
        if (done) break;
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The audit could not be completed.');
      setStatus('error');
      return { ok: false, message: caught instanceof Error ? caught.message : 'The audit could not be completed.' };
    } finally {
      setApiKey('');
      setShowKey(false);
    }
    return { ok: true };
  }

  async function runAudit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    await executeAudit(siteUrl, maxPages, provider, apiKey);
  }

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    void Promise.resolve(context.registerTool({
      name: 'run_recommendation_audit',
      title: 'Run recommendation audit',
      description: 'Audit a public ecommerce site and update the visible RecommendationAudit dashboard with deterministic, DOM-evidence-based findings.',
      inputSchema: {
        type: 'object',
        properties: {
          url: { type: 'string', description: 'Public HTTP or HTTPS ecommerce site URL.' },
          maxPages: { type: 'integer', minimum: 1, maximum: 20, default: 8 },
        },
        required: ['url'],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: true },
      async execute(input: unknown) {
        const value = input as { url?: unknown; maxPages?: unknown };
        if (typeof value?.url !== 'string' || !/^https?:\/\//i.test(value.url)) {
          throw new Error('url must be a public HTTP or HTTPS URL');
        }
        const pages = Math.min(20, Math.max(1, Number(value.maxPages) || 8));
        return executeAudit(value.url, pages, 'none', '');
      },
    }, { signal: lifecycle.signal })).catch(() => undefined);
    return () => lifecycle.abort();
  }, []);

  return (
    <main className="min-h-screen bg-[#f4f7f6] text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-5 py-4 lg:px-8">
          <div className="flex items-center gap-3">
            <span className="grid size-10 place-items-center rounded-xl bg-[#0b1828] text-emerald-400">
              <Radar className="size-5" />
            </span>
            <div>
              <p className="font-semibold tracking-tight">RecommendationAudit</p>
              <p className="text-xs text-slate-500">Presales intelligence workspace</p>
            </div>
          </div>
          <Badge className="hidden border-emerald-200 bg-emerald-50 text-emerald-700 sm:inline-flex" variant="outline">
            <ShieldCheck className="mr-1 size-3.5" /> Evidence-only analysis
          </Badge>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1500px] gap-6 px-5 py-6 lg:grid-cols-[340px_minmax(0,1fr)] lg:px-8 lg:py-8">
        <aside>
          <Card className="overflow-hidden border-0 bg-[#0b1828] text-white shadow-xl shadow-slate-900/10 lg:sticky lg:top-8">
            <CardHeader className="border-b border-white/10 px-6 pb-5">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-400">New audit</p>
              <CardTitle className="mt-2 text-xl text-white">Inspect a storefront</CardTitle>
              <p className="text-sm leading-6 text-slate-300">Map recommendation coverage and turn observed evidence into a focused sales story.</p>
            </CardHeader>
            <CardContent className="px-6 py-6">
              <form className="space-y-5" onSubmit={runAudit}>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-200" htmlFor="site-url">Ecommerce site URL</label>
                  <Input
                    className="h-11 border-white/15 bg-white/8 text-white placeholder:text-slate-500 focus-visible:border-emerald-400 focus-visible:ring-emerald-400/20"
                    id="site-url"
                    onChange={(event) => setSiteUrl(event.target.value)}
                    placeholder="https://store.example.com"
                    required
                    type="url"
                    value={siteUrl}
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-200" htmlFor="max-pages">Max pages</label>
                    <Input
                      className="h-10 border-white/15 bg-white/8 text-white"
                      id="max-pages"
                      max={20}
                      min={1}
                      onChange={(event) => setMaxPages(Number(event.target.value))}
                      type="number"
                      value={maxPages}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-200" htmlFor="provider">LLM analysis</label>
                    <NativeSelect
                      className="w-full [&_select]:h-10 [&_select]:border-white/15 [&_select]:bg-white/8 [&_select]:text-white"
                      id="provider"
                      onChange={(event) => setProvider(event.target.value)}
                      value={provider}
                    >
                      <NativeSelectOption value="none">Off</NativeSelectOption>
                      <NativeSelectOption value="openai">OpenAI</NativeSelectOption>
                    </NativeSelect>
                  </div>
                </div>

                {provider === 'openai' && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-200" htmlFor="api-key">OpenAI API key</label>
                    <div className="relative">
                      <Input
                        autoComplete="off"
                        className="h-11 border-white/15 bg-white/8 pr-11 text-white placeholder:text-slate-500"
                        id="api-key"
                        onChange={(event) => setApiKey(event.target.value)}
                        placeholder="sk-…"
                        required
                        type={showKey ? 'text' : 'password'}
                        value={apiKey}
                      />
                      <button
                        aria-label={showKey ? 'Hide API key' : 'Show API key'}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                        onClick={() => setShowKey((current) => !current)}
                        type="button"
                      >
                        {showKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                      </button>
                    </div>
                    <p className="flex gap-1.5 text-xs leading-5 text-slate-400"><LockKeyhole className="mt-0.5 size-3.5 shrink-0" />Used only for this run, then cleared.</p>
                  </div>
                )}

                <Button className="h-11 w-full bg-emerald-400 font-semibold text-[#071321] hover:bg-emerald-300" disabled={status === 'running'} type="submit">
                  {status === 'running' ? <><LoaderCircle className="animate-spin" /> Auditing storefront</> : <><FileSearch /> Run recommendation audit</>}
                </Button>
              </form>
            </CardContent>
          </Card>
        </aside>

        <section aria-live="polite" className="min-w-0">
          {status === 'idle' && (
            <Card className="min-h-[560px] place-items-center border-dashed border-slate-300 bg-white/70 px-6 text-center shadow-none">
              <CardContent className="max-w-lg py-16">
                <span className="mx-auto mb-6 grid size-16 place-items-center rounded-2xl bg-emerald-50 text-emerald-600"><Target className="size-8" /></span>
                <h1 className="text-3xl font-semibold tracking-tight text-slate-950">Make recommendation strategy visible.</h1>
                <p className="mt-4 text-base leading-7 text-slate-600">Enter a public ecommerce URL to discover recommendation placements, compare page-type coverage, and surface evidence-backed opportunities.</p>
                <div className="mt-8 grid gap-3 text-left sm:grid-cols-3">
                  {['DOM-based detection', 'Page-level evidence', 'Prioritized actions'].map((item, index) => (
                    <div className="rounded-xl border border-slate-200 bg-white p-4" key={item}>
                      <p className="text-xs font-semibold text-emerald-600">0{index + 1}</p>
                      <p className="mt-2 text-sm font-medium text-slate-700">{item}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {status === 'running' && (
            <Card className="min-h-[560px] justify-center border-slate-200 bg-white shadow-sm">
              <CardContent className="mx-auto w-full max-w-2xl py-16">
                <div className="flex items-center justify-between">
                  <div><p className="text-sm font-semibold text-emerald-600">Audit in progress</p><h1 className="mt-1 text-2xl font-semibold">Reading storefront evidence</h1></div>
                  <span className="font-mono text-2xl font-semibold text-slate-900">{progress.percent}%</span>
                </div>
                <Progress className="mt-6 [&_[data-slot=progress-indicator]]:bg-emerald-500 [&_[data-slot=progress-track]]:h-2" value={progress.percent} />
                <div className="mt-6 rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <p className="font-medium text-slate-800">{progress.message}</p>
                  {progress.currentUrl && <p className="mt-1 truncate text-sm text-slate-500">{progress.currentUrl}</p>}
                </div>
                <div className="mt-8 grid gap-3 sm:grid-cols-3">
                  {['Discover pages', 'Inspect DOM patterns', 'Score opportunities'].map((step, index) => {
                    const threshold = [10, 35, 75][index];
                    const active = progress.percent >= threshold;
                    return <div className={`rounded-xl border p-4 ${active ? 'border-emerald-200 bg-emerald-50' : 'border-slate-200'}`} key={step}><p className={`text-xs font-semibold ${active ? 'text-emerald-700' : 'text-slate-400'}`}>STEP {index + 1}</p><p className="mt-2 text-sm font-medium">{step}</p></div>;
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {status === 'error' && (
            <Card className="border-rose-200 bg-white shadow-sm">
              <CardContent className="py-8">
                <Alert className="border-rose-200 bg-rose-50 text-rose-800" variant="destructive">
                  <AlertCircle /><AlertTitle>Audit stopped</AlertTitle><AlertDescription>{error}</AlertDescription>
                </Alert>
                <Button className="mt-5" onClick={() => setStatus('idle')} variant="outline"><RefreshCw /> Try again</Button>
              </CardContent>
            </Card>
          )}

          {status === 'done' && result && (
            <div className="space-y-5">
              {result.warning && <Alert className="border-amber-200 bg-amber-50"><AlertCircle className="text-amber-600" /><AlertTitle>Analysis note</AlertTitle><AlertDescription>{result.warning}</AlertDescription></Alert>}

              <Card className="overflow-hidden border-0 bg-[#0b1828] text-white shadow-xl shadow-slate-900/10">
                <CardContent className="grid gap-7 p-6 md:grid-cols-[180px_1fr] md:p-8">
                  <div className="flex items-center gap-5 md:block">
                    <div className="grid size-32 shrink-0 place-items-center rounded-full p-2" style={{ background: `conic-gradient(#34d399 ${result.overallScore * 3.6}deg, #243247 0deg)` }}>
                      <div className="grid size-full place-items-center rounded-full bg-[#0b1828]"><div className="text-center"><strong className="text-4xl tracking-tight">{result.overallScore}</strong><p className="text-xs text-slate-400">out of 100</p></div></div>
                    </div>
                    <div className="md:mt-4"><p className="text-sm font-medium text-emerald-400">Overall score</p><p className="mt-1 text-sm text-slate-400">Evidence-based maturity snapshot</p></div>
                  </div>
                  <div>
                    <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-sm text-slate-400">Audit complete</p><h1 className="mt-1 text-2xl font-semibold tracking-tight">{new URL(result.siteUrl).hostname}</h1></div>{result.llmUsed && <Badge className="border-violet-400/30 bg-violet-400/10 text-violet-200" variant="outline"><Sparkles className="mr-1 size-3.5" /> OpenAI interpretation</Badge>}</div>
                    <div className="mt-7 grid grid-cols-3 divide-x divide-white/10 rounded-xl border border-white/10 bg-white/5 py-4 text-center">
                      <div><strong className="text-2xl">{result.pagesAnalyzed.length}</strong><p className="mt-1 text-xs text-slate-400">Pages analyzed</p></div>
                      <div><strong className="text-2xl">{result.placements.length}</strong><p className="mt-1 text-xs text-slate-400">Placements found</p></div>
                      <div><strong className="text-2xl">{coveredPageTypes}/{result.coverage.length}</strong><p className="mt-1 text-xs text-slate-400">Page types covered</p></div>
                    </div>
                    {result.summary && <p className="mt-5 border-l-2 border-emerald-400 pl-4 text-sm leading-6 text-slate-300">{result.summary}</p>}
                  </div>
                </CardContent>
              </Card>

              <div className="grid gap-5 xl:grid-cols-[1.05fr_.95fr]">
                <Card className="border-slate-200 bg-white shadow-sm">
                  <CardHeader><CardTitle><SectionTitle icon={<Globe2 className="size-4" />}>Recommendation coverage by page type</SectionTitle></CardTitle></CardHeader>
                  <CardContent className="space-y-5">
                    {result.coverage.map((item) => (
                      <div key={item.pageType}>
                        <div className="mb-2 flex justify-between text-sm"><span className="font-medium capitalize">{item.pageType}</span><span className="text-slate-500">{item.pagesWithPlacements}/{item.totalPages} pages · {item.percentage}%</span></div>
                        <div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-emerald-500" style={{ width: `${item.percentage}%` }} /></div>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                  <Card className="border-slate-200 bg-white shadow-sm">
                    <CardHeader><CardTitle><SectionTitle icon={<CheckCircle2 className="size-4" />}>Strengths</SectionTitle></CardTitle></CardHeader>
                    <CardContent className="space-y-4">{result.strengths.map((item) => <div key={item.title}><p className="text-sm font-semibold">{item.title}</p><p className="mt-1 text-sm leading-5 text-slate-600">{item.detail}</p><EvidenceLinks urls={item.evidenceUrls} /></div>)}</CardContent>
                  </Card>
                  <Card className="border-slate-200 bg-white shadow-sm">
                    <CardHeader><CardTitle><SectionTitle icon={<AlertCircle className="size-4" />}>Gaps</SectionTitle></CardTitle></CardHeader>
                    <CardContent className="space-y-4">{result.gaps.map((item) => <div key={item.title}><p className="text-sm font-semibold">{item.title}</p><p className="mt-1 text-sm leading-5 text-slate-600">{item.detail}</p><EvidenceLinks urls={item.evidenceUrls} /></div>)}</CardContent>
                  </Card>
                </div>
              </div>

              <Card className="border-slate-200 bg-white shadow-sm">
                <CardHeader><CardTitle><SectionTitle icon={<Lightbulb className="size-4" />}>Prioritized opportunities</SectionTitle></CardTitle></CardHeader>
                <CardContent className="grid gap-3 lg:grid-cols-3">
                  {result.opportunities.map((item) => <div className="rounded-xl border border-slate-200 p-4" key={item.title}><div className="flex items-start justify-between gap-3"><p className="font-semibold">{item.title}</p><PriorityBadge priority={item.priority} /></div><p className="mt-2 text-sm leading-6 text-slate-600">{item.detail}</p><EvidenceLinks urls={item.evidenceUrls} /></div>)}
                </CardContent>
              </Card>

              <Card className="border-slate-200 bg-white shadow-sm">
                <CardHeader><CardTitle><SectionTitle icon={<Radar className="size-4" />}>Detected placements</SectionTitle></CardTitle></CardHeader>
                <CardContent className="px-0">
                  {result.placements.length ? (
                    <Table>
                      <TableHeader><TableRow><TableHead className="pl-6">Placement</TableHead><TableHead>Page / position</TableHead><TableHead>Products</TableHead><TableHead>Confidence</TableHead><TableHead>Evidence</TableHead></TableRow></TableHeader>
                      <TableBody>{result.placements.map((placement, index) => <TableRow key={`${placement.pageUrl}-${placement.title}-${index}`}><TableCell className="max-w-64 pl-6"><p className="truncate font-medium">{placement.title}</p><a className="mt-1 block truncate text-xs text-slate-500 hover:text-emerald-600" href={placement.pageUrl} rel="noreferrer" target="_blank">{placement.pageUrl}</a></TableCell><TableCell><p className="capitalize">{placement.pageType}</p><p className="mt-1 text-xs text-slate-500">{placement.placementPosition}</p></TableCell><TableCell>{placement.productCount}</TableCell><TableCell><Badge className="border-emerald-200 bg-emerald-50 text-emerald-700" variant="outline">{Math.round(placement.confidence * 100)}%</Badge></TableCell><TableCell className="max-w-72 whitespace-normal text-xs leading-5 text-slate-500">{placement.evidence.slice(0, 2).join(' · ')}</TableCell></TableRow>)}</TableBody>
                    </Table>
                  ) : <p className="px-6 pb-6 text-sm text-slate-600">No qualifying recommendation placements were detected in the collected DOM evidence.</p>}
                </CardContent>
              </Card>

              <Card className="border-slate-200 bg-white shadow-sm">
                <CardHeader><CardTitle><SectionTitle icon={<ShieldCheck className="size-4" />}>Evidence URLs</SectionTitle></CardTitle></CardHeader>
                <CardContent className="grid gap-2 sm:grid-cols-2">
                  {result.evidenceUrls.map((url) => <a className="flex min-w-0 items-center justify-between gap-3 rounded-lg border border-slate-200 px-3 py-2.5 text-sm text-slate-600 hover:border-emerald-300 hover:text-emerald-700" href={url} key={url} rel="noreferrer" target="_blank"><span className="truncate">{url}</span><ArrowUpRight className="size-4 shrink-0" /></a>)}
                </CardContent>
              </Card>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
