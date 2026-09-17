/**
 * Runs sieval's post-edit checks after opencode edits a file.
 *
 * The checks live in scripts/post_edit_checks.py, shared with Claude Code.
 * This file only dispatches.
 *
 * Uses `file.edited` rather than `tool.execute.after`: its payload is a clean
 * `{ file: string }`, and opencode emits it from inside the write/edit/patch
 * tools, so it tracks agent edits rather than any file change on disk.
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
