#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";

export const PRODUCT_ID = "openclaw-local-plugin";
export const PRODUCT_PATH = "apps/memos-local-plugin";
export const PRODUCT_PATHS = [`${PRODUCT_PATH}/**`];
export const PRODUCT_TITLE = {
  zh: "OpenClaw 本地插件",
  en: "OpenClaw Local Plugin",
};
export const RELEASE_CATEGORY_ORDER = ["Added", "Improved", "Fixed"];
export const RELEASE_TO_DOC_CATEGORY = {
  Added: "New Features",
  Improved: "Improvements",
  Fixed: "Bug Fixes",
};
export const MAX_REPAIR_ATTEMPTS = 3;
const MAX_RELEASE_ITEMS = 12;
const MAX_TEXT_CN_CHARS = 180;
const MAX_TEXT_EN_CHARS = 220;
const CJK_RE = /[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]/;
const CJK_GLOBAL_RE = /[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]/g;
const TOKEN_RE =
  /(github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9_]+|npm_[A-Za-z0-9_]+|xox[baprs]-[A-Za-z0-9-]+|Bearer\s+[A-Za-z0-9._~+/=-]+)/g;
const INTERNAL_URL_RE =
  /https?:\/\/(?:(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|106\.15\.\d{1,3}\.\d{1,3})[^\s"'<>)]*/g;

export const RELEASE_NOTE_METHODS = [
  {
    source: "github-auto-generated-release-notes",
    url: "https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes",
    applied_as:
      "Keep the public MemOS Release body as GitHub-generated whole-repo What's Changed notes.",
  },
  {
    source: "github-generate-notes-api",
    url: "https://docs.github.com/en/rest/releases/releases?apiVersion=2022-11-28",
    applied_as:
      "Generate preview-only MemOS Release notes with previous_tag_name before creating a tag or GitHub Release.",
  },
  {
    source: "keep-a-changelog",
    url: "https://keepachangelog.com/en/1.1.0/",
    applied_as:
      "Write Plugin tab entries for humans, grouped by Added, Improved, and Fixed instead of dumping commits.",
  },
  {
    source: "conventional-commits",
    url: "https://www.conventionalcommits.org/en/v1.0.0/",
    applied_as:
      "Use commit type and scope as deterministic hints while requiring real product-path evidence.",
  },
  {
    source: "release-please",
    url: "https://github.com/googleapis/release-please",
    applied_as:
      "Treat feat, fix, perf, and refactor commits as releasable units and filter chore/docs/test noise.",
  },
];

function fail(message) {
  throw new Error(String(message));
}

function warn(message) {
  console.error(`::warning::${message}`);
}

function sh(args, options = {}) {
  return execFileSync("git", args, {
    cwd: process.cwd(),
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  }).trim();
}

function tryGit(args) {
  try {
    return sh(args);
  } catch {
    return "";
  }
}

export function redact(value) {
  return String(value ?? "")
    .replace(TOKEN_RE, "[REDACTED_TOKEN]")
    .replace(INTERNAL_URL_RE, "[REDACTED_INTERNAL_URL]")
    .replace(/([?&](?:token|access_token|secret|signature|service_id)=)[^&\s"')]+/gi, "$1[REDACTED]");
}

export function cleanVersion(raw) {
  const value = String(raw || "").trim();
  if (!value) return "";
  if (value.startsWith("v")) fail("version input must not include a leading v.");
  return value;
}

export function displayVersion(raw) {
  const value = cleanVersion(raw);
  return value ? `v${value}` : "";
}

export function parseSemver(raw) {
  const value = String(raw || "").trim().replace(/^v/, "");
  const match = value.match(/^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/);
  if (!match) return null;
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
    prerelease: match[4] ? match[4].split(".") : [],
  };
}

function compareIdentifier(a, b) {
  const aNum = /^\d+$/.test(a);
  const bNum = /^\d+$/.test(b);
  if (aNum && bNum) return Number(a) - Number(b);
  if (aNum) return -1;
  if (bNum) return 1;
  return a.localeCompare(b);
}

export function compareSemver(a, b) {
  const av = parseSemver(a);
  const bv = parseSemver(b);
  if (!av || !bv) return String(a).localeCompare(String(b));
  for (const key of ["major", "minor", "patch"]) {
    if (av[key] !== bv[key]) return av[key] - bv[key];
  }
  if (av.prerelease.length === 0 && bv.prerelease.length === 0) return 0;
  if (av.prerelease.length === 0) return 1;
  if (bv.prerelease.length === 0) return -1;
  const length = Math.max(av.prerelease.length, bv.prerelease.length);
  for (let index = 0; index < length; index += 1) {
    if (av.prerelease[index] === undefined) return -1;
    if (bv.prerelease[index] === undefined) return 1;
    const order = compareIdentifier(av.prerelease[index], bv.prerelease[index]);
    if (order !== 0) return order;
  }
  return 0;
}

export function validatePublishConfirmation({ dryRun, version, confirmation }) {
  if (String(dryRun) === "true") return;
  const expected = `PUBLISH v${cleanVersion(version)}`;
  if (String(confirmation || "").trim() !== expected) {
    fail(`dry_run=false requires publish_confirmation to exactly equal: ${expected}`);
  }
}

export function validateReleaseTarget({ dryRun, targetRef }) {
  if (String(dryRun) === "true") return;
  const value = String(targetRef || "main").trim();
  if (value !== "main") {
    fail("dry_run=false requires target_ref to be exactly main.");
  }
}

export function findPreviousMemOSTag(targetVersion, currentTag, tags) {
  const target = cleanVersion(targetVersion);
  const targetParsed = parseSemver(target);
  if (!targetParsed) fail(`Invalid semver version: ${targetVersion}`);
  const allowPrerelease = targetParsed.prerelease.length > 0;
  return tags
    .map((tag) => String(tag || "").trim())
    .filter((tag) => /^v\d+\.\d+\.\d+/.test(tag))
    .filter((tag) => tag !== currentTag)
    .map((tag) => ({ tag, version: tag.slice(1), parsed: parseSemver(tag) }))
    .filter((item) => item.parsed)
    .filter((item) => allowPrerelease || item.parsed.prerelease.length === 0)
    .filter((item) => compareSemver(item.version, target) < 0)
    .sort((a, b) => compareSemver(b.version, a.version))[0]?.tag || "";
}

function listTags() {
  return tryGit(["tag", "--list", "v*"])
    .split("\n")
    .map((tag) => tag.trim())
    .filter(Boolean);
}

function resolveRef(ref) {
  const value = String(ref || "HEAD").trim() || "HEAD";
  for (const candidate of [value, value.startsWith("origin/") ? "" : `origin/${value}`].filter(Boolean)) {
    const sha = tryGit(["rev-parse", "--verify", `${candidate}^{commit}`]);
    if (sha) return { ref: candidate, sha };
  }
  fail(`Cannot resolve target ref to a commit: ${value}`);
}

function gitShowJson(ref, path) {
  try {
    return JSON.parse(sh(["show", `${ref}:${path}`]));
  } catch {
    return {};
  }
}

function parseLines(text) {
  return String(text || "")
    .split("\n")
    .map((line) => line.trimEnd())
    .filter(Boolean);
}

export function sourceRefsFromText(text) {
  const refs = new Set();
  const value = String(text || "");
  const pattern = /\(#(\d+)\)|\b(?:PR|Fix(?:es)?|Close[sd]?|Refs?|Issue|in)\s+#(\d+)|\/(?:pull|issues)\/(\d+)\b/gi;
  for (const match of value.matchAll(pattern)) refs.add(`#${match[1] || match[2] || match[3]}`);
  return [...refs];
}

function extractPullRequests(commits, releaseAggregateItems, repo) {
  const seen = new Set();
  for (const commit of commits) {
    for (const ref of sourceRefsFromText(`${commit.subject || ""}\n${commit.body_excerpt || ""}`)) seen.add(ref.slice(1));
  }
  for (const item of releaseAggregateItems) {
    for (const ref of sourceRefsFromText(item.text)) seen.add(ref.slice(1));
  }
  return [...seen].sort((a, b) => Number(a) - Number(b)).map((number) => ({
    number,
    url: `https://github.com/${repo}/pull/${number}`,
  }));
}

function commitRefs(commit) {
  const refs = [];
  if (commit.short_sha) refs.push(commit.short_sha);
  if (commit.sha) refs.push(commit.sha);
  if (Array.isArray(commit.source_refs)) refs.push(...commit.source_refs);
  refs.push(...sourceRefsFromText(`${commit.subject || ""}\n${commit.body_excerpt || ""}`));
  return [...new Set(refs)];
}

function revertedCommitKeys(commits) {
  const keys = new Set();
  for (const commit of commits) {
    const text = `${commit.subject || ""}\n${commit.body_excerpt || ""}`;
    if (!/^revert\b/i.test(String(commit.subject || ""))) continue;
    const revertedSha = text.match(/This reverts commit ([0-9a-f]{7,40})\b/i)?.[1];
    if (revertedSha) keys.add(revertedSha);
    const revertedSubject = String(commit.subject || "").match(/^Revert\s+"(.+)"(?:\s+\(#\d+\))?$/i)?.[1];
    if (revertedSubject) keys.add(revertedSubject.toLowerCase());
  }
  return keys;
}

function isRevertedCommit(commit, revertedKeys) {
  if (!revertedKeys?.size) return false;
  const subject = String(commit.subject || "").toLowerCase();
  return [...revertedKeys].some((key) => {
    const value = String(key).toLowerCase();
    return commit.sha?.startsWith(value) || commit.short_sha?.startsWith(value) || subject === value;
  });
}

function isImportantCommit(commit, { revertedKeys = new Set() } = {}) {
  if (isRevertedCommit(commit, revertedKeys)) return false;
  const subject = String(commit.subject || "");
  if (/^merge\b/i.test(subject)) return false;
  if (/^(ci|chore|docs|test|style)(\([^)]+\))?:/i.test(subject)) return false;
  if (/^chore:\s*update version/i.test(subject)) return false;
  if (/^release:\s*merge\b/i.test(subject)) return true;
  if (/^revert\b/i.test(subject)) return false;
  return /^(feat|fix|perf|refactor|revert)(\([^)]+\))?:|add|improve|optimi[sz]e|compat|memory|plugin|openclaw|gateway|provider|hermes/i.test(subject);
}

function commitBodyExcerpt(sha) {
  const body = tryGit(["show", "--no-patch", "--format=%B", sha]);
  return redact(body).slice(0, 24000);
}

function releaseAggregateItems(commits) {
  const items = [];
  for (const commit of commits) {
    if (!/^release:\s*merge\b/i.test(String(commit.subject || ""))) continue;
    for (const rawLine of String(commit.body_excerpt || "").split("\n")) {
      const line = rawLine.trim();
      if (!/^\*\s+/.test(line)) continue;
      const text = line.replace(/^\*\s+/, "").trim();
      if (!text || text === commit.subject) continue;
      if (/^#\s*/.test(text)) continue;
      if (/^(co-authored-by|signed-off-by|---------|# conflicts:)/i.test(text)) continue;
      if (!/(#\d+|openclaw|memos-local|plugin|bridge|viewer|capture|reflection|llm|logging|openrouter|gateway|hermes|recovery|reward|episode|memory|provider)/i.test(text)) continue;
      const refs = sourceRefsFromText(text);
      if (!refs.length) continue;
      items.push({
        source_commit: commit.short_sha,
        text: redact(text),
        source_refs: [...new Set([commit.short_sha, ...refs])],
      });
      if (items.length >= 200) break;
    }
  }
  return items;
}

function evidenceCommitsForRelease(commits, aggregateItems) {
  const synthetic = aggregateItems
    .filter((item) => !/^revert\b/i.test(String(item.text || "")))
    .map((item) => {
      const prRefs = (item.source_refs || []).filter((ref) => String(ref).startsWith("#"));
      const sourceRefs = prRefs.length ? prRefs : [item.source_commit].filter(Boolean);
      return {
        sha: "",
        short_sha: "",
        subject: item.text,
        body_excerpt: "",
        source_refs: [...new Set(sourceRefs)],
        evidence_source: "release_aggregate_item",
      };
    });
  if (synthetic.length) return synthetic;
  return commits.filter((commit) => !/^revert\b/i.test(String(commit.subject || "")));
}

function packageChanges(previousTag, currentRef) {
  const path = `${PRODUCT_PATH}/package.json`;
  const before = gitShowJson(previousTag, path);
  const after = gitShowJson(currentRef, path);
  const fields = ["name", "version", "main", "types"];
  const changes = fields
    .filter((field) => before[field] !== after[field])
    .map((field) => ({ field, before: before[field], after: after[field] }));
  for (const section of ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"]) {
    const beforeDeps = before[section] || {};
    const afterDeps = after[section] || {};
    const names = new Set([...Object.keys(beforeDeps), ...Object.keys(afterDeps)]);
    for (const name of [...names].sort()) {
      if (beforeDeps[name] !== afterDeps[name]) {
        changes.push({ field: `${section}.${name}`, before: beforeDeps[name], after: afterDeps[name] });
      }
    }
  }
  return changes;
}

function collectPatchSnippets(range, changedFiles) {
  const interesting = changedFiles
    .map((item) => item.path)
    .filter((path) => /\.(ts|tsx|js|mjs|cjs|json|md|yaml|yml|sh|ps1)$/.test(path))
    .slice(0, 12);
  const snippets = [];
  let totalChars = 0;
  for (const path of interesting) {
    if (totalChars > 16000) break;
    const raw = tryGit(["diff", "--unified=1", "--no-ext-diff", range, "--", path]);
    if (!raw) continue;
    const text = redact(raw).slice(0, 5000);
    totalChars += text.length;
    snippets.push({ path, patch: text, truncated: raw.length > text.length });
  }
  return snippets;
}

export function collectLocalPluginEvidence({ previousTag, currentTag, currentRef, targetVersion, repo }) {
  const range = `${previousTag}..${currentRef}`;
  const commitText = tryGit([
    "log",
    "--format=%H%x09%h%x09%an%x09%ad%x09%s",
    "--date=iso-strict",
    range,
    "--",
    PRODUCT_PATH,
  ]);
  const commits = parseLines(commitText).map((line) => {
    const [sha = "", shortSha = "", author = "", date = "", subject = ""] = line.split("\t");
    const bodyExcerpt = commitBodyExcerpt(sha);
    const commit = {
      sha,
      short_sha: shortSha,
      author,
      date,
      subject: redact(subject),
      body_excerpt: bodyExcerpt,
    };
    return { ...commit, source_refs: commitRefs(commit) };
  });

  const changedFiles = parseLines(tryGit(["diff", "--name-status", range, "--", PRODUCT_PATH])).map((line) => {
    const parts = line.split("\t");
    const item = { status: parts[0], path: parts[parts.length - 1] };
    if (parts.length === 3) item.old_path = parts[1];
    return item;
  });

  const numstat = parseLines(tryGit(["diff", "--numstat", range, "--", PRODUCT_PATH])).map((line) => {
    const [additions = "0", deletions = "0", path = ""] = line.split("\t");
    return {
      path,
      additions: additions === "-" ? null : Number(additions),
      deletions: deletions === "-" ? null : Number(deletions),
    };
  });

  const aggregateItems = releaseAggregateItems(commits);
  const revertedKeys = revertedCommitKeys(commits);
  const evidenceCommits = evidenceCommitsForRelease(commits, aggregateItems);
  const importantCommits = evidenceCommits.filter((commit) => isImportantCommit(commit, { revertedKeys }));
  return {
    product_id: PRODUCT_ID,
    product_title: PRODUCT_TITLE,
    repo,
    release_repo: repo,
    previous_tag: previousTag,
    current_tag: currentTag,
    target_version: displayVersion(targetVersion),
    git_ref: currentRef,
    product_paths: PRODUCT_PATHS,
    has_product_changes: changedFiles.length > 0,
    commits: evidenceCommits,
    source_commits: commits,
    important_commits: importantCommits,
    release_aggregate_items: aggregateItems,
    reverted_change_keys: [...revertedKeys],
    required_source_refs: importantCommits.map((commit) => ({
      sha: commit.sha,
      short_sha: commit.short_sha,
      subject: commit.subject,
      accepted_refs: commitRefs(commit),
    })),
    pull_requests: extractPullRequests(commits, aggregateItems, repo),
    changed_files: changedFiles,
    diff_stat: {
      text: redact(tryGit(["diff", "--stat=200,200", range, "--", PRODUCT_PATH])),
      files: numstat,
    },
    important_diff: {
      [PRODUCT_PATHS[0]]: collectPatchSnippets(range, changedFiles),
    },
    package_changes: packageChanges(previousTag, currentRef),
    test_changes: changedFiles.filter((item) => /(^|\/)(test|tests|__tests__)\//.test(item.path) || /\.test\./.test(item.path)),
    docs_changes: changedFiles.filter((item) => /\.(md|mdx|rst)$/i.test(item.path)),
    release_note_quality_request: {
      candidate_count: 3,
      max_repair_attempts: MAX_REPAIR_ATTEMPTS,
      methodology: RELEASE_NOTE_METHODS,
      require_source_refs: true,
      require_bilingual_output: true,
      require_docs_preview: true,
      fail_closed: true,
      scoring: [
        "evidence coverage",
        "source_refs validity",
        "Chinese and English language purity",
        "Plugin tab readability",
      ],
      style_policy: [
        "Each bullet should explain the user-facing impact in one sentence.",
        "Avoid generic restatements such as '新增了 X 功能', '优化了 X 性能', or '修复了 X 问题'.",
        "A good bullet names the capability and says why it matters for OpenClaw local plugin users.",
      ],
      curation_policy: [
        "Use Conventional Commit type/scope as a hint, not as final copy.",
        "Group related commits or PR aggregate items into user-facing topics so Plugin tab output stays readable.",
        "Keep every covered source_ref in the draft and inspection artifact even when several commits become one bullet.",
        "Do not surface chore/docs/test-only noise unless it changes user-visible local plugin behavior.",
      ],
      example_rewrites: [
        {
          weak_cn: "优化了向量扫描性能。",
          better_cn: "优化批量向量扫描规划，降低大数据量本地同步时的处理压力。",
          weak_en: "Improved vector scan performance.",
          better_en: "Improved batch vector scan planning to reduce processing pressure during large local syncs.",
        },
      ],
    },
    target_surface: "memos_docs_plugin_changelog",
    release_context: {
      release_kind: "memos_whole_repo",
      public_release_body: "github_generated_whats_changed",
      docs_product_extraction: "path_filtered",
    },
    release_note_methodology: RELEASE_NOTE_METHODS,
  };
}

async function fetchJsonWithRetry(url, options, { label, attempts = 3, sleepMs = 500 } = {}) {
  const errors = [];
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, options);
      const text = await response.text();
      let payload = {};
      try {
        payload = text ? JSON.parse(text) : {};
      } catch {
        payload = { raw: text };
      }
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${JSON.stringify(payload).slice(0, 500)}`);
      }
      return payload;
    } catch (error) {
      errors.push(redact(error?.message || error));
      if (attempt === attempts) fail(`${label} failed after ${attempts} attempts: ${errors.join(" | ")}`);
      warn(`${label} attempt ${attempt}/${attempts} failed; retrying: ${errors[errors.length - 1]}`);
      await new Promise((resolve) => setTimeout(resolve, sleepMs * attempt));
    }
  }
  fail(`${label} failed.`);
}

export async function generateGitHubReleaseNotes({
  repo,
  currentTag,
  targetSha,
  previousTag,
  token = process.env.GITHUB_TOKEN || "",
}) {
  const localFallback = (warning = "") => {
    const subjects = parseLines(tryGit(["log", "--format=%s", `${previousTag}..${targetSha}`]));
    return {
      source: warning ? "local-fallback-after-github-error" : "local-fallback",
      name: `Release ${currentTag}`,
      body: [
        "## What's Changed",
        ...subjects.map((subject) => `* ${redact(subject)}`),
        "",
        `**Full Changelog**: https://github.com/${repo || "MemTensor/MemOS"}/compare/${previousTag}...${currentTag}`,
        "",
      ].join("\n"),
      warning,
    };
  };
  if (!token || !repo.includes("/")) {
    return localFallback(token ? "" : "GITHUB_TOKEN is not available; using local fallback release notes.");
  }

  try {
    const payload = await fetchJsonWithRetry(
      `https://api.github.com/repos/${repo}/releases/generate-notes`,
      {
        method: "POST",
        headers: {
          accept: "application/vnd.github+json",
          authorization: `Bearer ${token}`,
          "content-type": "application/json",
          "x-github-api-version": "2026-03-10",
        },
        body: JSON.stringify({
          tag_name: currentTag,
          target_commitish: targetSha,
          previous_tag_name: previousTag,
        }),
      },
      { label: "generate GitHub release notes" },
    );
    if (!String(payload.body || "").trim()) fail("GitHub generated release notes response was empty.");
    return {
      source: "github-generate-notes-api",
      name: String(payload.name || `Release ${currentTag}`),
      body: payload.body,
      warning: "",
    };
  } catch (error) {
    const message = redact(error?.message || error);
    const allowOffline = String(process.env.ALLOW_OFFLINE_DOCS_PREVIEW || "").toLowerCase() === "true";
    if (!allowOffline) throw error;
    warn(`GitHub generated release notes failed; using local fallback: ${message}`);
    return localFallback(message);
  }
}

const FALLBACK_TOPIC_RULES = [
  {
    pattern: /openrouter/i,
    category: "Added",
    text_cn: "**OpenRouter 提供商路由**：新增 OpenRouter 路由与 reasoning 配置支持，便于按配置选择模型提供商。",
    text_en: "**OpenRouter provider routing**: Added OpenRouter routing and reasoning configuration support for model selection.",
  },
  {
    pattern: /circuit breaker|terminal provider|insufficient balance|invalid api key/i,
    category: "Fixed",
    text_cn: "**LLM 熔断保护**：新增终端错误熔断，避免余额或密钥异常时持续触发后台 LLM 请求。",
    text_en: "**LLM circuit breaker**: Added terminal-error protection to stop repeated background LLM calls after billing or credential failures.",
  },
  {
    pattern: /recovery replay request storm|dirty-closed reward|reward recovery/i,
    category: "Fixed",
    text_cn: "**恢复任务稳定性**：优化脏关闭奖励恢复与回放分页，降低恢复过程中的请求风暴风险。",
    text_en: "**Recovery stability**: Improved dirty-closed reward recovery and replay pagination to reduce request storms.",
  },
  {
    pattern: /episode storm|foreground sessions|topic boundary|classifyTimeout|maxTurnsPerEpisode/i,
    category: "Fixed",
    text_cn: "**会话边界稳定性**：补强 episode 风暴保护和前台会话兜底，降低长任务阻塞风险。",
    text_en: "**Session-boundary stability**: Added episode-storm safeguards and foreground-session fallbacks to reduce long-task stalls.",
  },
  {
    pattern: /preserve v7 session defaults|v7-full-chain|default_config\.algorithm\.session|mergeMaxGapMs|followUpMode/i,
    category: "Fixed",
    text_cn: "**V7 会话默认配置**：保留默认 session 参数，避免自定义 follow-up 模式时丢失会话合并窗口。",
    text_en:
      "**V7 session defaults**: Preserved default session parameters so custom follow-up modes keep the merge window settings.",
  },
  {
    pattern: /captureRunner|reflectLlm|batch reflection|reflection scoring|chunk batch/i,
    category: "Improved",
    text_cn: "**采集反思稳定性**：优化批量反思评分与模型路由，降低长会话和 thinking 模型导致的解析风险。",
    text_en: "**Capture reflection stability**: Improved batch reflection scoring and model routing for long sessions and thinking-model setups.",
  },
  {
    pattern: /logging|timezone|memos\.log|logger/i,
    category: "Improved",
    text_cn: "**日志初始化与时区**：补齐本地桥接日志初始化和可配置时区，提升诊断一致性。",
    text_en: "**Logging initialization and timezone**: Added bridge logger initialization and configurable log timezone support.",
  },
  {
    pattern: /bridge|shutdown|session\.close|daemon|orphaned processes|rebuild|dist\/bridge|bridge\.cjs/i,
    category: "Fixed",
    text_cn: "**桥接进程稳定性**：增加会话关闭、shutdown 超时和桥接构建校验，减少事件循环阻塞、孤儿进程与旧产物风险。",
    text_en: "**Bridge process stability**: Added session-close, shutdown-timeout, and bridge rebuild safeguards to reduce event-loop blocking, orphaned processes, and stale artifacts.",
  },
  {
    pattern: /viewer|dashboard|metrics|namespace|500-row|overview/i,
    category: "Fixed",
    text_cn: "**Viewer 指标准确性**：修复命名空间切换和行数上限导致的概览统计偏差。",
    text_en: "**Viewer metric accuracy**: Fixed overview count drift caused by namespace switching and row-limit truncation.",
  },
  {
    pattern: /provider|llm config|embedding|model/i,
    category: "Improved",
    text_cn: "**模型配置与提供商兼容性**：优化 LLM、embedding 与 provider 配置处理，提升不同模型服务的接入稳定性。",
    text_en: "**Model configuration and provider compatibility**: Improved LLM, embedding, and provider configuration handling.",
  },
];

export function fallbackTopicForText(text, { allowGeneric = false } = {}) {
  const source = String(text || "");
  const rule = FALLBACK_TOPIC_RULES.find((item) => item.pattern.test(source));
  if (rule) return rule;
  if (!allowGeneric) return null;
  return {
    category: /^feat/i.test(source) ? "Added" : /^fix|^revert/i.test(source) ? "Fixed" : "Improved",
    text_cn: `**${PRODUCT_TITLE.zh}更新**：${source.replace(CJK_GLOBAL_RE, "").trim() || "本地插件能力完成更新。"}`,
    text_en: `**${PRODUCT_TITLE.en} update**: ${source.replace(CJK_GLOBAL_RE, "").trim() || "Release evidence updated."}`,
  };
}

function dedupeFallbackItems(items) {
  const byKey = new Map();
  for (const item of items) {
    const key = `${item.category}:${item.text_cn}:${item.text_en}`;
    if (!byKey.has(key)) {
      byKey.set(key, {
        ...item,
        source_refs: [...new Set(item.source_refs || [])],
      });
      continue;
    }
    const existing = byKey.get(key);
    existing.source_refs = [...new Set([...(existing.source_refs || []), ...(item.source_refs || [])])];
  }
  return [...byKey.values()];
}

function localFallbackDraft(evidence) {
  const revertedKeys = new Set((evidence.reverted_change_keys || []).map((item) => String(item).toLowerCase()));
  const aggregateItems = (evidence.release_aggregate_items || [])
    .filter((item) => !/^revert\b/i.test(item.text))
    .filter((item) => !revertedKeys.has(String(item.text || "").toLowerCase()));
  const sourceItems = aggregateItems.length
    ? aggregateItems.map((item) => {
        const prRefs = (item.source_refs || []).filter((ref) => String(ref).startsWith("#"));
        return {
          ...item,
          source_refs: prRefs.length ? prRefs : [item.source_commit].filter(Boolean),
        };
      })
    : evidence.important_commits.map((commit) => ({
        text: commit.subject,
        source_refs: [commit.short_sha],
      }));
  let items = dedupeFallbackItems(sourceItems
    .map((sourceItem) => {
      const topic = fallbackTopicForText(sourceItem.text, { allowGeneric: aggregateItems.length === 0 });
      if (!topic) return null;
      return {
        category: topic.category,
        text_cn: topic.text_cn,
        text_en: topic.text_en,
        source_refs: sourceItem.source_refs?.length ? sourceItem.source_refs : [evidence.important_commits[0]?.short_sha].filter(Boolean),
      };
    })
    .filter(Boolean)).slice(0, 10);
  if (!items.length && evidence.important_commits.length) {
    items = dedupeFallbackItems(evidence.important_commits.map((commit) => {
      const topic = fallbackTopicForText(commit.subject, { allowGeneric: true });
      return {
        category: topic.category,
        text_cn: topic.text_cn,
        text_en: topic.text_en,
        source_refs: [commit.short_sha],
      };
    })).slice(0, 10);
  }
  return {
    ok: true,
    needs_review: false,
    confidence: items.length ? "medium" : "high",
    warnings: ["offline fallback draft; use GitHub Actions with Doc Agent secrets for production quality"],
    release_items: items,
    coverage: {
      required_count: evidence.required_source_refs.length,
      covered_required_count: Math.min(items.length, evidence.required_source_refs.length),
      missing_required_count: Math.max(0, evidence.required_source_refs.length - items.length),
    },
    validation_attempt_count: 1,
    repair_attempt_count: 0,
  };
}

export function normalizeDraft(draft) {
  const releaseItems = Array.isArray(draft?.release_items) ? draft.release_items : [];
  return {
    ok: draft?.ok !== false,
    needs_review: Boolean(draft?.needs_review),
    confidence: draft?.confidence || "",
    warnings: Array.isArray(draft?.warnings) ? draft.warnings.map(redact) : [],
    coverage: draft?.coverage || {},
    release_items: releaseItems.map((item) => ({
      category: String(item.category || "").trim(),
      text_cn: String(item.text_cn || item.text || "").trim(),
      text_en: String(item.text_en || "").trim(),
      source_refs: Array.isArray(item.source_refs) ? item.source_refs.map((ref) => String(ref).trim()).filter(Boolean) : [],
    })),
    validation_attempt_count: Number(draft?.validation_attempt_count || 0),
    repair_attempt_count: Number(draft?.repair_attempt_count || 0),
  };
}

function stripBoldPrefix(text) {
  return String(text || "")
    .trim()
    .replace(/^\*\*[^*]+\*\*\s*[:：]\s*/, "")
    .trim();
}

function isGenericChineseDocsText(text) {
  const body = stripBoldPrefix(text).replace(/\s+/g, "");
  if (/(便于|降低|减少|避免|确保|支持|适配|稳定|同步|处理|接入|配置|演化|管道|压力|场景|大数据量)/.test(body)) {
    return false;
  }
  return /^(新增了|修复了|优化了|增加了).{1,40}(功能|问题|性能|能力)[。.]?$/.test(body);
}

export function validateDraft(draft, evidence) {
  const issues = [];
  const validRefs = new Set();
  for (const commit of evidence.commits || []) {
    for (const ref of commitRefs(commit)) validRefs.add(ref);
  }
  for (const pr of evidence.pull_requests || []) validRefs.add(`#${pr.number}`);

  if (!draft.ok) issues.push({ kind: "draft_not_ok", message: "draft ok=false" });
  if (draft.needs_review) issues.push({ kind: "needs_review", message: "draft needs review" });
  if (!draft.release_items.length && evidence.has_product_changes) {
    issues.push({ kind: "empty_release_items", message: "release_items is required when product files changed" });
  }
  if (draft.release_items.length > MAX_RELEASE_ITEMS) {
    issues.push({
      kind: "too_many_release_items",
      message: `release_items must be concise for the Plugin tab; got ${draft.release_items.length}, max ${MAX_RELEASE_ITEMS}`,
    });
  }

  for (const [index, item] of draft.release_items.entries()) {
    if (!RELEASE_CATEGORY_ORDER.includes(item.category)) {
      issues.push({ kind: "invalid_category", index, message: `invalid category ${item.category}` });
    }
    if (!item.text_cn || !CJK_RE.test(item.text_cn)) {
      issues.push({ kind: "invalid_text_cn", index, message: "text_cn must contain Chinese text" });
    }
    if (!item.text_en || CJK_RE.test(item.text_en)) {
      issues.push({ kind: "invalid_text_en", index, message: "text_en must be English without CJK characters" });
    }
    if (item.text_cn && item.text_cn.length > MAX_TEXT_CN_CHARS) {
      issues.push({
        kind: "text_cn_too_long",
        index,
        message: `text_cn is too long for docs rendering; got ${item.text_cn.length}, max ${MAX_TEXT_CN_CHARS}`,
      });
    }
    if (item.text_en && item.text_en.length > MAX_TEXT_EN_CHARS) {
      issues.push({
        kind: "text_en_too_long",
        index,
        message: `text_en is too long for docs rendering; got ${item.text_en.length}, max ${MAX_TEXT_EN_CHARS}`,
      });
    }
    if (isGenericChineseDocsText(item.text_cn)) {
      issues.push({
        kind: "generic_text_cn",
        index,
        message: "text_cn restates the change too generically; include concrete user-facing impact.",
      });
    }
    if (!item.source_refs.length) {
      issues.push({ kind: "missing_source_refs", index, message: "source_refs is required" });
    }
    for (const ref of item.source_refs) {
      if (!validRefs.has(ref)) {
        issues.push({ kind: "invalid_source_ref", index, ref, message: `source_ref does not match evidence: ${ref}` });
      }
    }
  }

  const coveredRefs = new Set(draft.release_items.flatMap((item) => item.source_refs));
  const missingRequired = [];
  for (const required of evidence.required_source_refs || []) {
    if (!required.accepted_refs.some((ref) => coveredRefs.has(ref))) {
      missingRequired.push(required.short_sha);
    }
  }
  for (const ref of missingRequired) {
    issues.push({ kind: "missing_required_ref", ref, message: `important commit is not covered: ${ref}` });
  }

  return {
    ok: issues.length === 0,
    needs_review: issues.length > 0,
    issue_count: issues.length,
    issues,
    coverage: {
      required_count: evidence.required_source_refs?.length || 0,
      covered_required_count: (evidence.required_source_refs?.length || 0) - missingRequired.length,
      missing_required_count: missingRequired.length,
      missing_required_refs: missingRequired,
    },
  };
}

async function requestDocAgentDraft(evidence) {
  if (!evidence.has_product_changes) {
    return {
      ok: true,
      needs_review: false,
      confidence: "high",
      warnings: ["No OpenClaw local plugin path changes in this MemOS release range."],
      release_items: [],
      coverage: { required_count: 0, covered_required_count: 0, missing_required_count: 0 },
      validation_attempt_count: 1,
      repair_attempt_count: 0,
    };
  }

  const url = String(process.env.DOC_AGENT_RELEASE_NOTES_DRAFT_URL || "").trim();
  const token = String(process.env.DOC_AGENT_RELEASE_NOTES_DRAFT_TOKEN || "").trim();
  const allowOffline = String(process.env.ALLOW_OFFLINE_DOCS_PREVIEW || "").toLowerCase() === "true";
  if ((!url || !token) && allowOffline) return normalizeDraft(localFallbackDraft(evidence));
  if (!url) fail("DOC_AGENT_RELEASE_NOTES_DRAFT_URL secret is required for MemOS release docs preview.");
  if (!token) fail("DOC_AGENT_RELEASE_NOTES_DRAFT_TOKEN secret is required for MemOS release docs preview.");

  const attempts = [];
  let repairContext = null;
  for (let attempt = 1; attempt <= MAX_REPAIR_ATTEMPTS; attempt += 1) {
    const payload = await fetchJsonWithRetry(
      url,
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          ...evidence,
          workflow_retry_context: {
            attempt,
            previous_errors: attempts,
          },
          repair_context: repairContext,
        }),
      },
      { label: "Doc Agent local-plugin docs draft" },
    );
    const draft = normalizeDraft(payload);
    const validation = validateDraft(draft, evidence);
    attempts.push({ attempt, validation });
    if (validation.ok) {
      return {
        ...draft,
        validation_report: validation,
        validation_attempt_count: attempt,
        repair_attempt_count: attempt - 1,
        coverage: validation.coverage,
      };
    }
    repairContext = {
      validation_report: validation,
      instructions: [
        "Repair only the validation issues.",
        "Keep facts within the provided evidence.",
        "Return release_items with category, text_cn, text_en, and source_refs.",
        "For generic_text_cn issues, explain concrete user-facing impact without inventing facts.",
      ],
    };
  }
  fail(`Doc Agent draft failed validation after ${MAX_REPAIR_ATTEMPTS} attempts: ${JSON.stringify(attempts.at(-1)?.validation?.issues || [])}`);
}

export function buildDocsPreview(draft, evidence) {
  const makeSide = (language) => {
    const categories = {};
    for (const releaseCategory of RELEASE_CATEGORY_ORDER) {
      const docsCategory = RELEASE_TO_DOC_CATEGORY[releaseCategory];
      const changedInfo = draft.release_items
        .filter((item) => item.category === releaseCategory)
        .map((item) => (language === "zh" ? item.text_cn : item.text_en))
        .filter(Boolean);
      if (changedInfo.length) {
        categories[docsCategory] = [
          {
            type: language === "zh" ? PRODUCT_TITLE.zh : PRODUCT_TITLE.en,
            changedInfo,
          },
        ];
      }
    }
    return {
      name: evidence.current_tag,
      source: {
        repo: evidence.repo,
        tag: evidence.current_tag,
        release_url: `https://github.com/${evidence.repo}/releases/tag/${evidence.current_tag}`,
        previous_tag: evidence.previous_tag,
        product_paths: evidence.product_paths,
      },
      products: {
        plugin: categories,
      },
    };
  };
  return { cn: makeSide("zh"), en: makeSide("en") };
}

export function docsPreviewMarkdown(preview, draft, evidence) {
  const lines = [
    `# ${PRODUCT_TITLE.zh}-${evidence.current_tag}`,
    "",
    `- source: ${evidence.repo}`,
    `- range: ${evidence.previous_tag}...${evidence.current_tag}`,
    `- product_paths: ${evidence.product_paths.join(", ")}`,
    "",
  ];
  if (!evidence.has_product_changes) {
    lines.push("No OpenClaw local plugin changes were found in this MemOS release range.", "");
    return lines.join("\n");
  }
  for (const [language, label, field] of [
    ["cn", "中文", "text_cn"],
    ["en", "English", "text_en"],
  ]) {
    lines.push(`## ${label}`, "");
    for (const releaseCategory of RELEASE_CATEGORY_ORDER) {
      const items = draft.release_items.filter((item) => item.category === releaseCategory);
      if (!items.length) continue;
      lines.push(`### ${RELEASE_TO_DOC_CATEGORY[releaseCategory]}`, "");
      for (const item of items) {
        lines.push(`- ${item[field]}`);
      }
      lines.push("");
    }
    if (!Object.keys(preview[language].products.plugin).length) lines.push("No entries.", "");
  }
  lines.push("## Source Refs", "");
  for (const item of draft.release_items) {
    lines.push(`- ${item.category}: ${item.source_refs.join(", ")}`);
  }
  lines.push("");
  return lines.join("\n");
}

function appendOutput(name, value) {
  if (!process.env.GITHUB_OUTPUT) return;
  const text = String(value ?? "");
  writeFileSync(process.env.GITHUB_OUTPUT, `${name}<<EOF\n${text}\nEOF\n`, { flag: "a" });
}

function writeJson(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export async function run() {
  const version = cleanVersion(process.env.RELEASE_VERSION);
  if (!version) fail("RELEASE_VERSION is required.");
  if (!parseSemver(version)) fail(`Invalid semver version: ${version}`);

  const dryRun = String(process.env.DRY_RUN ?? "true");
  validatePublishConfirmation({
    dryRun,
    version,
    confirmation: process.env.PUBLISH_CONFIRMATION || "",
  });

  const currentTag = `v${version}`;
  const repo = process.env.GITHUB_REPOSITORY || "MemTensor/MemOS";
  const targetRefInput = process.env.TARGET_REF || "main";
  validateReleaseTarget({ dryRun, targetRef: targetRefInput });
  const target = resolveRef(targetRefInput);
  const previousTag = process.env.PREVIOUS_TAG || findPreviousMemOSTag(version, currentTag, listTags());
  if (!previousTag) fail(`Cannot find previous MemOS v* tag before ${currentTag}.`);

  const releaseNotes = await generateGitHubReleaseNotes({
    repo,
    currentTag,
    targetSha: target.sha,
    previousTag,
  });
  const evidence = collectLocalPluginEvidence({
    previousTag,
    currentTag,
    currentRef: target.sha,
    targetVersion: version,
    repo,
  });
  evidence.memos_release_notes = {
    source: releaseNotes.source,
    name: releaseNotes.name,
    body_preview: redact(releaseNotes.body).slice(0, 12000),
  };

  const draft = await requestDocAgentDraft(evidence);
  const validation = validateDraft(draft, evidence);
  if (!validation.ok) fail(`Validated draft is not acceptable: ${JSON.stringify(validation.issues)}`);

  const preview = buildDocsPreview(draft, evidence);
  const outputRoot =
    process.env.INSPECTION_DIR ||
    join(tmpdir(), `memos-release-${currentTag.replace(/[^A-Za-z0-9_.-]/g, "-")}-inspection`);
  mkdirSync(outputRoot, { recursive: true });

  const releaseNotesFile = join(outputRoot, "memos-release-notes.md");
  const releaseNotesAliasFile = join(outputRoot, "release-notes.md");
  const evidenceFile = join(outputRoot, "local-plugin-evidence.json");
  const evidenceAliasFile = join(outputRoot, "evidence.json");
  const draftFile = join(outputRoot, "local-plugin-docs-draft.json");
  const docsPreviewFile = join(outputRoot, "local-plugin-docs-preview.json");
  const docsPreviewAliasFile = join(outputRoot, "docs-preview.json");
  const docsPreviewMarkdownFile = join(outputRoot, "local-plugin-docs-preview.md");
  const docsPreviewMarkdownAliasFile = join(outputRoot, "docs-preview.md");
  const qualityReportFile = join(outputRoot, "quality-report.json");
  const readmeFile = join(outputRoot, "README.md");

  writeFileSync(releaseNotesFile, `${releaseNotes.body.trim()}\n`, "utf8");
  writeFileSync(releaseNotesAliasFile, `${releaseNotes.body.trim()}\n`, "utf8");
  const redactedEvidence = JSON.parse(redact(JSON.stringify(evidence, null, 2)));
  writeJson(evidenceFile, redactedEvidence);
  writeJson(evidenceAliasFile, redactedEvidence);
  writeJson(draftFile, draft);
  writeJson(docsPreviewFile, preview);
  writeJson(docsPreviewAliasFile, preview);
  writeFileSync(docsPreviewMarkdownFile, docsPreviewMarkdown(preview, draft, evidence), "utf8");
  writeFileSync(docsPreviewMarkdownAliasFile, docsPreviewMarkdown(preview, draft, evidence), "utf8");
  const qualityReport = {
    ok: validation.ok,
    source_id: PRODUCT_ID,
    release_kind: "memos_whole_repo",
    docs_product_extraction: "path_filtered",
    public_release_body: "github_generated_whats_changed",
    dry_run: dryRun === "true",
    release_notes_source: releaseNotes.source,
    current_tag: currentTag,
    previous_tag: previousTag,
    target_ref: target.ref,
    target_sha: target.sha,
    product_paths: evidence.product_paths,
    has_product_changes: evidence.has_product_changes,
    changed_file_count: evidence.changed_files.length,
    commit_count: evidence.commits.length,
    important_commit_count: evidence.important_commits.length,
    release_item_count: draft.release_items.length,
    release_note_methodology: RELEASE_NOTE_METHODS,
    coverage: validation.coverage,
    validation_report: validation,
    validation_attempt_count: draft.validation_attempt_count,
    repair_attempt_count: draft.repair_attempt_count,
    warnings: draft.warnings,
    docs_preview_files: ["content/cn/plugin-changelog.yml", "content/en/plugin-changelog.yml"],
    no_side_effects: {
      npm_publish: false,
      oss_upload: false,
      production_docs_pr: false,
      pre_gray_production: false,
    },
  };
  writeJson(qualityReportFile, qualityReport);
  writeFileSync(
    readmeFile,
    [
      "# MemOS release inspection",
      "",
      `- source_id: ${PRODUCT_ID}`,
      "- release_kind: memos_whole_repo",
      "- docs_product_extraction: path_filtered",
      "- public_release_body: github_generated_whats_changed",
      `- dry_run: ${dryRun}`,
      `- current_tag: ${currentTag}`,
      `- previous_tag: ${previousTag}`,
      `- target_ref: ${target.ref}`,
      `- target_sha: ${target.sha}`,
      `- product_paths: ${evidence.product_paths.join(", ")}`,
      `- release_notes_source: ${releaseNotes.source}`,
      `- has_product_changes: ${evidence.has_product_changes}`,
      `- validation_attempt_count: ${draft.validation_attempt_count}`,
      `- repair_attempt_count: ${draft.repair_attempt_count}`,
      "- no_side_effects: npm_publish=false, oss_upload=false, production_docs_pr=false, pre_gray_production=false",
      "",
      "Files:",
      "",
      "- memos-release-notes.md",
      "- release-notes.md",
      "- local-plugin-evidence.json",
      "- evidence.json",
      "- local-plugin-docs-draft.json",
      "- local-plugin-docs-preview.md",
      "- local-plugin-docs-preview.json",
      "- docs-preview.md",
      "- docs-preview.json",
      "- quality-report.json",
      "",
    ].join("\n"),
    "utf8",
  );

  appendOutput("inspection_dir", outputRoot);
  appendOutput("memos_release_notes_file", releaseNotesFile);
  appendOutput("release_notes_file", releaseNotesFile);
  appendOutput("evidence_file", evidenceFile);
  appendOutput("docs_preview_file", docsPreviewFile);
  appendOutput("docs_preview_markdown_file", docsPreviewMarkdownFile);
  appendOutput("quality_report_file", qualityReportFile);
  appendOutput("source_id", PRODUCT_ID);
  appendOutput("previous_tag", previousTag);
  appendOutput("current_tag", currentTag);
  appendOutput("target_ref", target.ref);
  appendOutput("target_sha", target.sha);
  appendOutput("has_product_changes", String(evidence.has_product_changes));
  appendOutput("release_notes_source", releaseNotes.source);
  appendOutput("validation_attempt_count", String(draft.validation_attempt_count ?? ""));
  appendOutput("repair_attempt_count", String(draft.repair_attempt_count ?? ""));

  console.log(`Prepared MemOS release inspection in ${outputRoot}`);
  console.log(`Release notes source: ${releaseNotes.source}`);
  console.log(`Range: ${previousTag}..${target.sha}`);
  console.log(`OpenClaw local plugin changed files: ${evidence.changed_files.length}`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  run().catch((error) => {
    console.error(`::error::${redact(error?.stack || error?.message || error)}`);
    process.exit(1);
  });
}
