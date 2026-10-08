const state = {
  books: [],
  activeBookSlug: null,
  sharedIndexAvailable: false,
  pending: false,
  mapOpen: false,
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
  if (state.sharedIndexAvailable) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `book-button${state.activeBookSlug === "__all__" ? " active" : ""}`;
    button.innerHTML = `
      <div class="book-title">All books</div>
      <div class="book-meta">Shared library query</div>
    `;
    button.addEventListener("click", () => selectBook("__all__"));
    bookList.appendChild(button);
  }

  for (const book of state.books) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `book-button${book.slug === state.activeBookSlug ? " active" : ""}`;
    button.innerHTML = `
      <div class="book-author">${escapeHtml(book.author || "")}</div>
      <div class="book-title">${escapeHtml(book.display_title || book.title)}</div>
      <div class="book-meta">${book.chunk_count ?? "?"} chunks</div>
    `;
    button.addEventListener("click", () => selectBook(book.slug));
    bookList.appendChild(button);
  }
}

function selectBook(bookSlug) {
  state.activeBookSlug = bookSlug;
  const book = getActiveBook();
  const isAllBooks = bookSlug === "__all__";
  activeBookTitle.textContent = isAllBooks ? "All indexed books" : (book ? bookLabel(book) : "Choose a book");
  composerMeta.textContent = isAllBooks
    ? "Ready to query the shared library"
    : (book ? `Ready to query ${bookLabel(book)}` : "No book selected");
  renderBooks();
  if (state.mapOpen) {
    loadCampaign();
  }
}

const mapToggle = document.getElementById("map-toggle");
const mapView = document.getElementById("map-view");
const mapMessage = document.getElementById("map-message");
const mapSettings = document.getElementById("map-settings");
let leafletMap = null;
let campaignLayer = null;
let campaignRequestId = 0;

function setMapOpen(open) {
  state.mapOpen = open;
  document.body.classList.toggle("map-mode", open);
  mapView.hidden = !open;
  mapSettings.hidden = !open;
  mapToggle.textContent = open ? "Chat" : "Map";
  if (!open) {
    return;
  }
  if (!leafletMap) {
    leafletMap = L.map("map").setView([44.5, 16.5], 6);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(leafletMap);
    campaignLayer = L.layerGroup().addTo(leafletMap);
  }
  leafletMap.invalidateSize();
  loadCampaign();
}

function showMapMessage(text) {
  mapMessage.textContent = text;
  mapMessage.hidden = !text;
}

// Catmull-Rom spline through the points, sampled into a dense polyline.
function splinePath(points, steps = 12) {
  if (points.length < 3) {
    return points;
  }
  const path = [];
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(i - 1, 0)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(i + 2, points.length - 1)];
    for (let s = 0; s < steps; s++) {
      const t = s / steps;
      const t2 = t * t;
      const t3 = t2 * t;
      path.push(p1.map((_, k) =>
        0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * t +
          (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
          (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)));
    }
  }
  path.push(points[points.length - 1]);
  return path;
}

async function loadCampaign(focusFileIndex = null) {
  const requestId = ++campaignRequestId;
  campaignLayer.clearLayers();
  showMapMessage("");
  const book = getActiveBook();
  if (!book) {
    showMapMessage(state.activeBookSlug === "__all__"
      ? "Campaign maps are available for a single book. Select a book title."
      : "Select a book to see its campaign.");
    return;
  }
  const variant = document.querySelector('input[name="campaign-variant"]:checked').value;
  let data;
  try {
    const response = await fetch(`/api/campaign/${encodeURIComponent(book.slug)}?variant=${variant}`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `Request failed (${response.status})`);
    }
    data = payload;
  } catch (error) {
    if (requestId === campaignRequestId) {
      showMapMessage(`No campaign to display: ${error.message}`);
    }
    return;
  }
  if (requestId !== campaignRequestId) {
    return;
  }

  const latLngs = data.points.map((point) => [point.lat, point.lng]);
  const unique = latLngs.filter((p, i) => i === 0 || p[0] !== latLngs[i - 1][0] || p[1] !== latLngs[i - 1][1]);
  if (unique.length > 1) {
    L.polyline(splinePath(unique), { color: "#b3412c", weight: 3, opacity: 0.8 }).addTo(campaignLayer);
  }
  const markers = [];
  let focusMarker = null;
  const goToPoint = (index) => {
    leafletMap.panTo(markers[index].getLatLng());
    markers[index].openPopup();
  };
  data.points.forEach((point, index) => {
    const isFirst = index === 0;
    const marker = L.circleMarker([point.lat, point.lng], {
      radius: isFirst ? 10 : 7,
      color: isFirst ? "#0b3d91" : "#7a2a1c",
      weight: 2,
      fillColor: isFirst ? "#2f7be5" : "#e0674d",
      fillOpacity: 0.95,
    });
    const content = document.createElement("div");
    content.innerHTML =
      `<strong>${index + 1}. ${escapeHtml(point.operation || "(no operation)")}</strong><br>` +
      `Date: ${escapeHtml(point.date || "unknown")}<br>Place: ${escapeHtml(point.place || "unknown")}` +
      `<div class="popup-nav"><button type="button" class="popup-prev" aria-label="Previous point">&larr;</button>` +
      `<span>${index + 1} / ${data.points.length}</span>` +
      `<button type="button" class="popup-next" aria-label="Next point">&rarr;</button></div>`;
    const prev = content.querySelector(".popup-prev");
    const next = content.querySelector(".popup-next");
    prev.disabled = index === 0;
    next.disabled = index === data.points.length - 1;
    prev.addEventListener("click", () => goToPoint(index - 1));
    next.addEventListener("click", () => goToPoint(index + 1));
    const editButton = document.createElement("button");
    editButton.type = "button";
    editButton.className = "popup-edit";
    editButton.textContent = "Edit";
    editButton.addEventListener("click", () =>
      openEditDialog(book.slug, data.variant, point, () => loadCampaign(point.index)));
    content.prepend(editButton);
    marker.bindPopup(content).addTo(campaignLayer);
    markers.push(marker);
    if (point.index === focusFileIndex) {
      focusMarker = marker;
    }
  });
  leafletMap.fitBounds(L.latLngBounds(latLngs).pad(0.2), { maxZoom: 11 });
  if (focusMarker) {
    leafletMap.panTo(focusMarker.getLatLng());
    focusMarker.openPopup();
  }
}

function openEditDialog(bookSlug, variant, point, onSaved) {
  const entry = structuredClone(point.entry);
  const overlay = document.createElement("div");
  overlay.className = "edit-overlay";
  const form = document.createElement("form");
  form.className = "edit-dialog";
  form.innerHTML = "<h3>Edit campaign entry</h3>";

  const inputs = [];
  const addField = (label, value, setter, kind) => {
    const wrapper = document.createElement("label");
    wrapper.className = "edit-field";
    wrapper.append(document.createTextNode(label));
    const long = kind === "text" && (String(value).length > 80 || label === "notes");
    const input = document.createElement(long ? "textarea" : "input");
    if (long) {
      input.rows = 5;
    } else if (kind === "number") {
      input.type = "number";
      input.step = "any";
      input.required = true;
    } else {
      input.type = "text";
    }
    input.value = value ?? "";
    wrapper.appendChild(input);
    form.appendChild(wrapper);
    inputs.push(() => setter(kind === "number" ? Number(input.value) : input.value));
  };

  for (const [key, value] of Object.entries(entry)) {
    if (key === "coordinates") {
      const coords = value && typeof value === "object" ? value : {};
      entry.coordinates = coords;
      addField("latitude", coords.lat, (v) => { coords.lat = v; }, "number");
      addField("longitude", coords.lng, (v) => { coords.lng = v; }, "number");
    } else if (value === null || typeof value === "string") {
      addField(key, value, (v) => { entry[key] = v; }, "text");
    } else if (typeof value === "number") {
      addField(key, value, (v) => { entry[key] = v; }, "number");
    }
  }

  const error = document.createElement("div");
  error.className = "edit-error";
  const actions = document.createElement("div");
  actions.className = "edit-actions";
  actions.innerHTML =
    '<button type="button" class="edit-cancel">Cancel</button><button type="submit" class="edit-save">Save</button>';
  form.append(error, actions);
  overlay.appendChild(form);
  document.body.appendChild(overlay);

  const close = () => overlay.remove();
  actions.querySelector(".edit-cancel").addEventListener("click", close);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    inputs.forEach((apply) => apply());
    const saveButton = actions.querySelector(".edit-save");
    saveButton.disabled = true;
    try {
      const response = await fetch(
        `/api/campaign/${encodeURIComponent(bookSlug)}?variant=${variant}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ index: point.index, entry }),
        });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(typeof payload.detail === "string" ? payload.detail : `Request failed (${response.status})`);
      }
      close();
      onSaved();
    } catch (err) {
      error.textContent = `Save failed: ${err.message}`;
      saveButton.disabled = false;
    }
  });
}

mapToggle.addEventListener("click", () => setMapOpen(!state.mapOpen));
document.querySelectorAll('input[name="campaign-variant"]').forEach((input) =>
  input.addEventListener("change", () => loadCampaign()));

function bookLabel(book) {
  return book.display_title || book.title;
}

// Order by the first unit number in the title; unnumbered books go last, then alphabetical.
function sortBooks(books) {
  const unitNumber = (book) => {
    const match = bookLabel(book).match(/\d+/);
    return match ? Number(match[0]) : Infinity;
  };
  return [...books].sort((a, b) =>
    (unitNumber(a) - unitNumber(b) || 0) ||
    bookLabel(a).localeCompare(bookLabel(b), "sr-Latn"));
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

  state.books = sortBooks(payload.books || []);
  state.sharedIndexAvailable = Boolean(payload.shared_index_available);
  renderBooks();
  if (state.sharedIndexAvailable) {
    selectBook("__all__");
    setStatus("Ready");
  } else if (state.books.length > 0) {
    const indexedBook = state.books.find((book) => book.has_book_index);
    selectBook((indexedBook || state.books[0]).slug);
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
  const activeBookSlug = state.activeBookSlug;
  const isAllBooks = activeBookSlug === "__all__";

  if ((!book && !isAllBooks) || !question || state.pending) {
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
        book_slug: activeBookSlug,
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

    const answeredBook = state.books.find((item) => item.slug === payload.book.slug);
    const answeredTitle = answeredBook ? bookLabel(answeredBook) : payload.book.title;
    const metaLine = payload.provider
      ? `${answeredTitle} | ${payload.provider} | ${payload.model} | ${payload.timing_ms} ms`
      : `${answeredTitle} | ${payload.mode} | ${payload.timing_ms} ms`;
    const chunkLine = Array.isArray(payload.chunk_refs) && payload.chunk_refs.length > 0
      ? `chunks: ${payload.chunk_refs.join(", ")}`
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
