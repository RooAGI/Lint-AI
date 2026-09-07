#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="examples/demo-real-terminal"
QUERY='How many retry attempts should we use for gateway timeouts?'
BIN='./target/release/lint-ai'

prompt() {
  printf '\n\033[1;36m$ %s\033[0m\n' "$1"
  sleep 0.7
}

run_cmd() {
  prompt "$1"
  bash -lc "$1"
  sleep 1.0
}

clear
printf '\033[1;37mLint-AI — real terminal recording\033[0m\n'
printf 'No supersedes metadata. Same policy domain. Newer file wins by chronology.\n'
sleep 1.2

run_cmd "cat $DEMO_DIR/decision-a.md"
run_cmd "cat $DEMO_DIR/decision-b.md"
run_cmd "stat -c '%n  %y' $DEMO_DIR/decision-a.md $DEMO_DIR/decision-b.md | sed 's/\\.000000000 +0000//'"

prompt "$BIN --query \"$QUERY\" $DEMO_DIR | jq '{result: (.results[0] | {doc_id, semantic_status}), aggregation}'"
"$BIN" --query "$QUERY" "$DEMO_DIR" \
  | jq '{result: (.results[0] | {doc_id, semantic_status}), aggregation}'
sleep 1.4

prompt "$BIN --llm-context \"$QUERY\" --result-count 5 $DEMO_DIR | jq '{current_context: ([.top_chunks[] | select(.doc_id == \"decision-b.md\")][0] | {doc_id, text}), stale_value_in_context: any(.top_chunks[]; (.text // \"\") | contains(\"retry attempts: 5\"))}'"
"$BIN" --llm-context "$QUERY" --result-count 5 "$DEMO_DIR" \
  | jq '{current_context: ([.top_chunks[] | select(.doc_id == "decision-b.md")][0] | {doc_id, text}), stale_value_in_context: any(.top_chunks[]; (.text // "") | contains("retry attempts: 5"))}'
sleep 1.8

printf '\n\033[1;32mResult: the current value is 2; stale value 5 is excluded from agent context.\033[0m\n'
printf '\033[2mRecorded live from the actual lint-ai CLI.\033[0m\n'
sleep 2
