/**
 * Runs sieval's post-edit checks after opencode edits a file.
 *
 * The checks themselves live in scripts/post_edit_checks.py, shared with
 * Claude Code and Codex. This file only dispatches.
 *
 * Uses the `file.edited` event rather than `tool.execute.after`: its payload
 * is `{ file: string }`, a clean path, where the tool hook would require
 * per-tool argument parsing to find one.
 *
 * AI-Generated Code - Claude Sonnet 5 (Anthropic)
 */
export const PostEditChecks = async ({ $, directory }) => {
  return {
    event: async ({ event }) => {
      if (event.type !== "file.edited") return
      const file = event.properties?.file
      if (!file) return
      await $`python ${directory}/scripts/post_edit_checks.py ${file}`.nothrow()
    },
  }
}
