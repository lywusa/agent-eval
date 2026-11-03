"""All policy prompts. Frozen at pre-registration (week 3)."""

REACT_SYSTEM = """You are a research assistant that answers multi-hop questions using a Wikipedia-paragraph search tool.

Work step by step. In every turn call exactly ONE tool:
- search(query): keyword search; returns doc_ids, titles and snippets.
- read(doc_id): returns the full paragraph of one document.
- finish(answer, cited_doc_ids): give the final answer and the doc_ids you read that support it.

Rules:
- Base your answer on documents you have read, and cite them.
- The answer must be a short phrase (an entity, name, date, number, or yes/no), not a sentence.
- You have at most {max_steps} tool calls. Call finish before you run out."""

REACT_USER = "Question: {question}"

# Used when the backend has no native tool-calling chat template (e.g. the mock backend
# or a non-Qwen model); Qwen3's template appends an equivalent block automatically.
TOOL_FORMAT_FALLBACK = """

To call a tool, output a JSON object inside <tool_call></tool_call> tags, for example:
<tool_call>
{"name": "search", "arguments": {"query": "..."}}
</tool_call>"""

PARSE_ERROR_FEEDBACK = (
    "ERROR: could not parse a tool call. Respond with exactly one tool call in the form "
    '<tool_call>{"name": "<search|read|finish>", "arguments": {...}}</tool_call>.'
)

LAST_STEP_NUDGE = "This is your last tool call. Call finish now with your best answer and citations."

CLOSED_BOOK_SYSTEM = """Answer the question from your own knowledge. You have no tools.
Reply in exactly this format:
Answer: <short phrase>"""

SINGLE_RAG_SYSTEM = """Answer the question using the retrieved documents below.
Reply in exactly this format:
Answer: <short phrase>
Citations: <comma-separated doc_ids of the documents that support the answer>"""

SINGLE_RAG_USER = """Retrieved documents:
{documents}

Question: {question}"""

# Appended when a thinking-mode generation exhausts its thinking budget, following
# Qwen's documented early-exit recipe.
THINKING_EARLY_EXIT = (
    "\n\nConsidering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n</think>\n\n"
)
