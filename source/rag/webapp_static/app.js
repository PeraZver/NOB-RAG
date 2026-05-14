const state = {
  books: [],
  activeBookSlug: null,
  pending: false,
};

const bookList = document.getElementById("book-list");
const activeBookTitle = document.getElementById("active-book-title");
const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const sendButton = document.getElementById("send-button");
const statusPill = document.getElementById("status-pill");
const composerMeta = document.getElementById("composer-meta");
const providerSelect = document.getElementById("provider-select");
const profileSelect = document.getElementById("profile-select");
const retrieveOnlyInput = document.getElementById("retrieve-only");
const messageTemplate = document.getElementById("message-template");

function setStatus(text) {
  statusPill.textContent = text;
}

function getActiveBook() {
  return state.books.find((book) => book.slug === state.activeBookSlug) || null;
}

function renderBooks() {
  bookList.innerHTML = "";
  for (const book of state.books) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `book-button${book.slug === state.activeBookSlug ? " active" : ""}`;
    button.innerHTML = `
      <div class="book-title">${book.title}</div>
      <div class="book-meta">${book.chunk_count ?? "?"} chunks</div>
    `;
    button.addEventListener("click", () => selectBook(book.slug));
    bookList.appendChild(button);
  }
}

function selectBook(bookSlug) {
  state.activeBookSlug = bookSlug;
  const book = getActiveBook();
  activeBookTitle.textContent = book ? book.title : "Choose a book";
  composerMeta.textContent = book ? `Ready to query ${book.title}` : "No book selected";
  renderBooks();
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInlineMarkdown(text) {
  let rendered = escapeHtml(text);
  rendered = rendered.replace(/`([^`]+)`/g, "<code>$1</code>");
  rendered = rendered.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  rendered = rendered.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  return rendered;
}

function renderMarkdown(markdown) {
  const codeBlocks = [];
  let working = markdown.replace(/```([a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g, (_match, language, code) => {
    const block = `<pre><code${language ? ` class="language-${escapeHtml(language)}"` : ""}>${escapeHtml(code.trimEnd())}</code></pre>`;
    const token = `__CODE_BLOCK_${codeBlocks.length}__`;
    codeBlocks.push(block);
    return token;
  });

  const lines = working.split(/\r?\n/);
  const parts = [];
  let paragraphLines = [];
  let listItems = [];
  let listType = null;

  function flushParagraph() {
    if (paragraphLines.length === 0) {
      return;
    }
    const paragraph = paragraphLines.join("<br>");
    parts.push(`<p>${paragraph}</p>`);
    paragraphLines = [];
  }

  function flushList() {
    if (listItems.length === 0 || !listType) {
      return;
    }
    const tag = listType === "ol" ? "ol" : "ul";
    parts.push(`<${tag}>${listItems.map((item) => `<li>${item}</li>`).join("")}</${tag}>`);
    listItems = [];
    listType = null;
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    if (/^__CODE_BLOCK_\d+__$/.test(trimmed)) {
      flushParagraph();
      flushList();
      parts.push(trimmed);
      continue;
    }

    const headingMatch = /^(#{1,3})\s+(.*)$/.exec(trimmed);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = headingMatch[1].length;
      parts.push(`<h${level + 2}>${renderInlineMarkdown(headingMatch[2])}</h${level + 2}>`);
      continue;
    }

    const unorderedMatch = /^[-*]\s+(.*)$/.exec(trimmed);
    if (unorderedMatch) {
      flushParagraph();
      if (listType && listType !== "ul") {
        flushList();
      }
      listType = "ul";
      listItems.push(renderInlineMarkdown(unorderedMatch[1]));
      continue;
    }

    const orderedMatch = /^\d+\.\s+(.*)$/.exec(trimmed);
    if (orderedMatch) {
      flushParagraph();
      if (listType && listType !== "ol") {
        flushList();
      }
      listType = "ol";
      listItems.push(renderInlineMarkdown(orderedMatch[1]));
      continue;
    }

    flushList();
    paragraphLines.push(renderInlineMarkdown(trimmed));
  }

  flushParagraph();
  flushList();

  let rendered = parts.join("");
  codeBlocks.forEach((block, index) => {
    rendered = rendered.replace(`__CODE_BLOCK_${index}__`, block);
  });
  return rendered;
}

function appendMessage(role, body) {
  const fragment = messageTemplate.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  article.classList.add(role);
  fragment.querySelector(".message-role").textContent = role === "user" ? "You" : "Assistant";
  const messageBody = fragment.querySelector(".message-body");
  if (role === "assistant") {
    messageBody.innerHTML = renderMarkdown(body);
  } else {
    messageBody.textContent = body;
  }

  chatLog.appendChild(fragment);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setPending(isPending) {
  state.pending = isPending;
  sendButton.disabled = isPending;
  questionInput.disabled = isPending;
  setStatus(isPending ? "Thinking..." : "Idle");
}

async function parseResponsePayload(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return await response.json();
  }

  const text = await response.text();
  return { detail: text || `HTTP ${response.status}` };
}

async function loadBooks() {
  setStatus("Loading books...");
  const response = await fetch("/api/books");
  const payload = await parseResponsePayload(response);
  if (!response.ok) {
    throw new Error(payload.detail || "Failed to load books.");
  }

  state.books = payload.books || [];
  renderBooks();
  if (state.books.length > 0) {
    selectBook(state.books[0].slug);
    setStatus("Ready");
  } else {
    activeBookTitle.textContent = "No indexed books found";
    composerMeta.textContent = "Create chunks.jsonl and chroma_db for at least one book.";
    setStatus("No books");
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  const book = getActiveBook();

  if (!book || !question || state.pending) {
    return;
  }

  appendMessage("user", question);
  questionInput.value = "";
  setPending(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        book_slug: book.slug,
        question,
        provider: providerSelect.value,
        profile: profileSelect.value,
        retrieve_only: retrieveOnlyInput.checked,
      }),
    });

    const payload = await parseResponsePayload(response);
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed.");
    }

    const metaLine = payload.provider
      ? `${payload.book.title} | ${payload.provider} | ${payload.model} | ${payload.timing_ms} ms`
      : `${payload.book.title} | ${payload.mode} | ${payload.timing_ms} ms`;
    const chunkLine = Array.isArray(payload.chunk_ids) && payload.chunk_ids.length > 0
      ? `chunk_ids: ${payload.chunk_ids.join(", ")}`
      : "";
    const messageBody = chunkLine
      ? `${payload.answer}\n\n${metaLine}\n${chunkLine}`
      : `${payload.answer}\n\n${metaLine}`;
    appendMessage("assistant", messageBody);
    setStatus("Ready");
  } catch (error) {
    appendMessage("assistant", `Error: ${error.message}`);
    setStatus("Error");
  } finally {
    setPending(false);
    questionInput.focus();
  }
});

loadBooks().catch((error) => {
  appendMessage("assistant", `Error loading books: ${error.message}`);
  setStatus("Error");
});
