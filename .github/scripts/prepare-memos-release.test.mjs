import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  PRODUCT_ID,
  RELEASE_NOTE_METHODS,
  buildDocsPreview,
  compareSemver,
  cleanVersion,
  docsPreviewMarkdown,
  fallbackTopicForText,
  findPreviousMemOSTag,
  generateGitHubReleaseNotes,
  sourceRefsFromText,
  validateDraft,
  validatePublishConfirmation,
  validateReleaseTarget,
} from "./prepare-memos-release.mjs";

const evidence = {
  repo: "MemTensor/MemOS",
  previous_tag: "v2.0.24",
  current_tag: "v2.0.25",
  product_paths: ["apps/memos-local-plugin/**"],
  has_product_changes: true,
  commits: [
    {
      sha: "9deb941e00000000000000000000000000000000",
      short_sha: "9deb941e",
      subject: "feat(l3): dedicated l3Llm config slot for abstraction pass (#1959)",
    },
    {
      sha: "59c1474600000000000000000000000000000000",
      short_sha: "59c14746",
      subject: "Fix #2076: local-plugin gateway CPU 100% - synchronous full-table vector scan (#2077)",
    },
  ],
  important_commits: [
    {
      sha: "9deb941e00000000000000000000000000000000",
      short_sha: "9deb941e",
      subject: "feat(l3): dedicated l3Llm config slot for abstraction pass (#1959)",
    },
    {
      sha: "59c1474600000000000000000000000000000000",
      short_sha: "59c14746",
      subject: "Fix #2076: local-plugin gateway CPU 100% - synchronous full-table vector scan (#2077)",
    },
  ],
  required_source_refs: [
    {
      short_sha: "9deb941e",
      accepted_refs: ["9deb941e", "9deb941e00000000000000000000000000000000", "#1959"],
    },
    {
      short_sha: "59c14746",
      accepted_refs: ["59c14746", "59c1474600000000000000000000000000000000", "#2076", "#2077"],
    },
  ],
  pull_requests: [{ number: "1959" }, { number: "2076" }, { number: "2077" }],
};

const validDraft = {
  ok: true,
  needs_review: false,
  release_items: [
    {
      category: "Added",
      text_cn: "**L3 抽象模型配置**：新增专用 L3 LLM 配置入口，便于独立管理抽象结论阶段的模型调用。",
      text_en: "**L3 abstraction model configuration**: Added a dedicated L3 LLM configuration entry for the abstraction pass.",
      source_refs: ["9deb941e"],
    },
    {
      category: "Improved",
      text_cn: "**向量扫描性能优化**：优化本地插件网关的大批量向量扫描流程，降低同步全表扫描造成的 CPU 压力。",
      text_en: "**Vector scan performance**: Optimized large local-plugin vector scans to reduce CPU pressure from synchronous full-table reads.",
      source_refs: ["59c14746", "#2077"],
    },
  ],
};

test("compares prerelease versions with SemVer precedence", () => {
  assert.ok(compareSemver("1.0.0-beta.10", "1.0.0-beta.9") > 0);
  assert.ok(compareSemver("1.0.0-beta.20", "1.0.0-beta.19") > 0);
  assert.ok(compareSemver("1.0.0", "1.0.0-beta.20") > 0);
  assert.equal(compareSemver("1.0.0+build.2", "1.0.0+build.1"), 0);
});

test("selects the previous MemOS stable tag for release evidence", () => {
  assert.equal(
    findPreviousMemOSTag("2.0.25", "v2.0.25", ["v2.0.24", "v2.0.25", "v2.0.25-beta.1", "memos-local-plugin-v2.0.10"]),
    "v2.0.24",
  );
  assert.equal(
    findPreviousMemOSTag("2.0.26-beta.2", "v2.0.26-beta.2", ["v2.0.25", "v2.0.26-beta.1", "v2.0.24"]),
    "v2.0.26-beta.1",
  );
});

test("extracts PR refs from GitHub release note wording", () => {
  assert.deepEqual(
    sourceRefsFromText(
      "feat: add provider routing by @someone in #1958\nFix #2131: dashboard drift (#2132)\nhttps://github.com/MemTensor/MemOS/pull/2146",
    ),
    ["#1958", "#2131", "#2132", "#2146"],
  );
});

test("rejects leading v in manual version input", () => {
  assert.equal(cleanVersion("2.0.25"), "2.0.25");
  assert.throws(() => cleanVersion("v2.0.25"), /must not include a leading v/);
});

test("requires an exact publish confirmation for non-dry-run releases", () => {
  assert.doesNotThrow(() => validatePublishConfirmation({ dryRun: "true", version: "2.0.25", confirmation: "" }));
  assert.throws(
    () => validatePublishConfirmation({ dryRun: "false", version: "2.0.25", confirmation: "" }),
    /PUBLISH v2\.0\.25/,
  );
  assert.doesNotThrow(() =>
    validatePublishConfirmation({ dryRun: "false", version: "2.0.25", confirmation: "PUBLISH v2.0.25" }),
  );
});

test("publish workflow defaults real releases to draft before release.published", () => {
  const workflow = readFileSync(".github/workflows/memos-release-publish.yml", "utf8");
  assert.match(workflow, /create_draft_release:/);
  assert.match(workflow, /default:\s+true/);
  assert.match(workflow, /CREATE_DRAFT_RELEASE/);
  assert.match(workflow, /flags\+=\(--draft\)/);
  assert.match(workflow, /Publish manually to trigger release\.published/);
});

test("inspection artifact contract includes generic aliases and side-effect proof", () => {
  const script = readFileSync(".github/scripts/prepare-memos-release.mjs", "utf8");
  assert.match(script, /"release-notes\.md"/);
  assert.match(script, /"evidence\.json"/);
  assert.match(script, /"docs-preview\.md"/);
  assert.match(script, /"docs-preview\.json"/);
  assert.match(script, /source_id:\s+PRODUCT_ID/);
  assert.match(script, /release_kind:\s+"memos_whole_repo"/);
  assert.match(script, /docs_product_extraction:\s+"path_filtered"/);
  assert.match(script, /public_release_body:\s+"github_generated_whats_changed"/);
  assert.match(script, /no_side_effects:\s+\{/);
  assert.match(script, /npm_publish:\s+false/);
  assert.match(script, /production_docs_pr:\s+false/);
  assert.equal(PRODUCT_ID, "openclaw-local-plugin");
});

test("allows flexible target refs only for dry runs", () => {
  assert.doesNotThrow(() => validateReleaseTarget({ dryRun: "true", targetRef: "origin/main" }));
  assert.doesNotThrow(() => validateReleaseTarget({ dryRun: "false", targetRef: "main" }));
  assert.throws(() => validateReleaseTarget({ dryRun: "false", targetRef: "origin/main" }), /exactly main/);
  assert.throws(() => validateReleaseTarget({ dryRun: "false", targetRef: "feature/test" }), /exactly main/);
});

test("validates a bilingual source-referenced plugin docs draft", () => {
  const result = validateDraft(validDraft, evidence);
  assert.equal(result.ok, true);
  assert.equal(result.coverage.required_count, 2);
  assert.equal(result.coverage.covered_required_count, 2);
});

test("release note methodology records the sources used for quality policy", () => {
  assert.ok(RELEASE_NOTE_METHODS.some((item) => item.source === "github-auto-generated-release-notes"));
  assert.ok(RELEASE_NOTE_METHODS.some((item) => item.source === "keep-a-changelog"));
  assert.ok(RELEASE_NOTE_METHODS.some((item) => item.source === "conventional-commits"));
  assert.ok(RELEASE_NOTE_METHODS.every((item) => item.url.startsWith("https://")));
});

test("fallback topic rewrites V7 session default fixes into user-facing docs copy", () => {
  const topic = fallbackTopicForText("fix(plugin): preserve V7 session defaults (#2158)", { allowGeneric: true });
  assert.equal(topic.category, "Fixed");
  assert.match(topic.text_cn, /V7 会话默认配置/);
  assert.match(topic.text_cn, /会话合并窗口/);
  assert.match(topic.text_en, /V7 session defaults/);
  assert.doesNotMatch(topic.text_cn, /fix\(plugin\)/);
});

test("GitHub release notes fallback stays whole-repo when API access is unavailable", async () => {
  const result = await generateGitHubReleaseNotes({
    repo: "MemTensor/MemOS",
    currentTag: "v0.0.0-test",
    targetSha: "HEAD",
    previousTag: "HEAD",
    token: "",
  });
  assert.equal(result.source, "local-fallback-after-github-error");
  assert.match(result.body, /## What's Changed/);
  assert.match(result.body, /Full Changelog/);
  assert.doesNotMatch(result.body, /source_refs/);
  assert.doesNotMatch(result.body, /doc-agent-release-notes-json/);
});

test("rejects English text that still contains Chinese", () => {
  const result = validateDraft(
    {
      ...validDraft,
      release_items: [{ ...validDraft.release_items[0], text_en: "Added L3 抽象 model configuration." }],
    },
    evidence,
  );
  assert.equal(result.ok, false);
  assert.ok(result.issues.some((issue) => issue.kind === "invalid_text_en"));
});

test("rejects missing or invented source refs", () => {
  const missing = validateDraft(
    {
      ...validDraft,
      release_items: [{ ...validDraft.release_items[0], source_refs: [] }, validDraft.release_items[1]],
    },
    evidence,
  );
  assert.equal(missing.ok, false);
  assert.ok(missing.issues.some((issue) => issue.kind === "missing_source_refs"));

  const invented = validateDraft(
    {
      ...validDraft,
      release_items: [{ ...validDraft.release_items[0], source_refs: ["deadbee"] }, validDraft.release_items[1]],
    },
    evidence,
  );
  assert.equal(invented.ok, false);
  assert.ok(invented.issues.some((issue) => issue.kind === "invalid_source_ref"));
});

test("rejects drafts that drop important commits", () => {
  const result = validateDraft({ ...validDraft, release_items: [validDraft.release_items[0]] }, evidence);
  assert.equal(result.ok, false);
  assert.ok(result.issues.some((issue) => issue.kind === "missing_required_ref" && issue.ref === "59c14746"));
});

test("rejects plugin docs drafts that are too fragmented for the changelog page", () => {
  const noisyItems = Array.from({ length: 13 }, (_item, index) => ({
    category: "Improved",
    text_cn: `**本地插件优化 ${index + 1}**：整理发布说明展示效果。`,
    text_en: `**Local plugin improvement ${index + 1}**: Refined release-note presentation.`,
    source_refs: [index % 2 === 0 ? "9deb941e" : "59c14746"],
  }));
  const result = validateDraft({ ...validDraft, release_items: noisyItems }, evidence);
  assert.equal(result.ok, false);
  assert.ok(result.issues.some((issue) => issue.kind === "too_many_release_items"));
});

test("rejects plugin docs bullets that are too long to render well", () => {
  const result = validateDraft(
    {
      ...validDraft,
      release_items: [
        {
          ...validDraft.release_items[0],
          text_cn: `**L3 抽象模型配置**：${"用于发布说明质量验证的重复中文描述。".repeat(12)}`,
          text_en: `**L3 abstraction model configuration**: ${"This repeated English detail is intentionally too verbose for a changelog bullet. ".repeat(6)}`,
        },
        validDraft.release_items[1],
      ],
    },
    evidence,
  );
  assert.equal(result.ok, false);
  assert.ok(result.issues.some((issue) => issue.kind === "text_cn_too_long"));
  assert.ok(result.issues.some((issue) => issue.kind === "text_en_too_long"));
});

test("rejects generic Chinese plugin docs copy", () => {
  const result = validateDraft(
    {
      ...validDraft,
      release_items: [
        {
          ...validDraft.release_items[0],
          text_cn: "**本地插件能力**：新增了本地插件能力功能。",
        },
        validDraft.release_items[1],
      ],
    },
    evidence,
  );
  assert.equal(result.ok, false);
  assert.ok(result.issues.some((issue) => issue.kind === "generic_text_cn"));
});

test("rejects generic English plugin docs copy", () => {
  const result = validateDraft(
    {
      ...validDraft,
      release_items: [
        {
          ...validDraft.release_items[0],
          text_en: "**Local plugin update**: Fixed local plugin issue.",
        },
        validDraft.release_items[1],
      ],
    },
    evidence,
  );
  assert.equal(result.ok, false);
  assert.ok(result.issues.some((issue) => issue.kind === "generic_text_en"));
});

test("rejects raw Conventional Commit subjects copied into docs copy", () => {
  const result = validateDraft(
    {
      ...validDraft,
      release_items: [
        {
          ...validDraft.release_items[0],
          text_en: "**V7 defaults**: fix(plugin): preserve V7 session defaults (#2158).",
        },
        validDraft.release_items[1],
      ],
    },
    evidence,
  );
  assert.equal(result.ok, false);
  assert.ok(result.issues.some((issue) => issue.kind === "raw_commit_subject_text" && issue.field === "text_en"));
});

test("rejects duplicate plugin docs bullets that should be merged", () => {
  const result = validateDraft(
    {
      ...validDraft,
      release_items: [
        validDraft.release_items[0],
        {
          ...validDraft.release_items[0],
          source_refs: ["59c14746"],
        },
      ],
    },
    evidence,
  );
  assert.equal(result.ok, false);
  assert.ok(result.issues.some((issue) => issue.kind === "duplicate_release_item"));
});

test("accepts concise impact-oriented Chinese plugin docs copy", () => {
  const result = validateDraft(
    {
      ...validDraft,
      release_items: [
        validDraft.release_items[0],
        {
          ...validDraft.release_items[1],
          text_cn: "**向量扫描性能优化**：优化了自适应向量扫描批处理，提升了大数据量同步时的处理效率。",
        },
      ],
    },
    evidence,
  );
  assert.equal(result.ok, true);
});

test("builds Plugin tab previews without exposing source refs in page content", () => {
  const preview = buildDocsPreview(validDraft, evidence);
  assert.equal(preview.source_id, "openclaw-local-plugin");
  assert.equal(preview.source_repo, "MemTensor/MemOS");
  assert.equal(preview.previous_tag, "v2.0.24");
  assert.equal(preview.current_tag, "v2.0.25");
  assert.equal(preview.would_create_docs_pr, false);
  assert.deepEqual(preview.files, ["content/cn/plugin-changelog.yml", "content/en/plugin-changelog.yml"]);
  assert.equal(preview.cn.name, "v2.0.25");
  assert.equal(preview.cn.source.repo, "MemTensor/MemOS");
  assert.deepEqual(preview.cn.source.product_paths, ["apps/memos-local-plugin/**"]);
  assert.equal(preview.cn.products.plugin["New Features"][0].type, "OpenClaw 本地插件");
  assert.equal(preview.en.products.plugin.Improvements[0].type, "OpenClaw Local Plugin");

  const markdown = docsPreviewMarkdown(preview, validDraft, evidence);
  assert.match(markdown, /Source Refs/);
  assert.match(markdown, /9deb941e/);
  assert.match(markdown, /59c14746/);
});
