const state = {
  documents: [],
  conversations: [],
  messages: [],
  currentSection: "ask",
  preferredMode: "global",
  documentSearchQuery: "",
  documentFilter: "all",
  activeDocumentId: null,
  activeConversationId: null,
  isSubmitting: false,
  pendingImages: [],
  pdfPanel: {
    visible: false,
    documentId: null,
    filename: "",
    pdfUrl: "",
    page: 1,
    width: 680,
    loading: false,
    error: "",
  },
}

const refs = {}
let workspaceRenderQueued = false
let mermaidRenderQueued = false
let shouldAutoScroll = true
let pendingWaitTickerId = null
const activeOcrPolls = new Set()
const AUTO_SCROLL_THRESHOLD = 48
const PDF_DRAWER_MIN_WIDTH = 560
const PDF_DRAWER_MAX_WIDTH = 960
const PAGE_REF_PATTERN = /第\s*(\d+)\s*(?:[–\-~至]\s*(\d+)\s*)?页|\[(\d+)(?:\s*[–\-~至]\s*(\d+))?\]|[Pp](?:age)?\.?\s*(\d+)(?:\s*[–\-~]\s*(\d+))?/g

function persistStateToHash() {
  const parts = []
  if (state.currentSection === "documents") {
    parts.push("section=documents")
  }
  if (state.activeConversationId) {
    parts.push(`conv=${state.activeConversationId}`)
  } else if (state.activeDocumentId) {
    parts.push(`doc=${state.activeDocumentId}`)
  }
  const hash = parts.length ? `#${parts.join("&")}` : ""
  if (window.location.hash !== hash && hash) {
    window.history.replaceState(null, "", hash)
  }
}

function restoreStateFromHash() {
  const hash = window.location.hash.slice(1)
  if (!hash) {
    return {}
  }
  const params = new URLSearchParams(hash)
  return {
    section: params.get("section") || "ask",
    conversationId: params.get("conv") ? Number(params.get("conv")) : null,
    documentId: params.get("doc") ? Number(params.get("doc")) : null,
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  bindRefs()
  bindEvents()
  configureMarkdown()
  renderPdfDrawer()
  await bootstrap()
})

function bindRefs() {
  refs.workspace = document.getElementById("workspace")
  refs.brandButton = document.getElementById("brandButton")
  refs.navAskButton = document.getElementById("navAskButton")
  refs.navDocumentsButton = document.getElementById("navDocumentsButton")
  refs.uploadButton = document.getElementById("uploadButton")
  refs.docsUploadButton = document.getElementById("docsUploadButton")
  refs.sidebarStatusMeta = document.getElementById("sidebarStatusMeta")
  refs.fileInput = document.getElementById("fileInput")
  refs.documentMeta = document.getElementById("documentMeta")
  refs.documentLibraryGrid = document.getElementById("documentLibraryGrid")
  refs.documentSearchInput = document.getElementById("documentSearchInput")
  refs.documentFilterBar = document.getElementById("documentFilterBar")
  refs.jumpToDocumentsButton = document.getElementById("jumpToDocumentsButton")
  refs.globalChatButton = document.getElementById("globalChatButton")
  refs.singleChatButton = document.getElementById("singleChatButton")
  refs.conversationMeta = document.getElementById("conversationMeta")
  refs.conversationLibraryList = document.getElementById("conversationLibraryList")
  refs.conversationTitle = document.getElementById("conversationTitle")
  refs.headerBreadcrumb = document.getElementById("headerBreadcrumb")
  refs.headerMeta = document.getElementById("headerMeta")
  refs.currentDocumentPill = document.getElementById("currentDocumentPill")
  refs.askView = document.getElementById("askView")
  refs.documentsView = document.getElementById("documentsView")
  refs.askModeTag = document.getElementById("askModeTag")
  refs.askScopeText = document.getElementById("askScopeText")
  refs.statusBanner = document.getElementById("statusBanner")
  refs.statusText = document.getElementById("statusText")
  refs.statusActionButton = document.getElementById("statusActionButton")
  refs.emptyTitle = document.getElementById("emptyTitle")
  refs.emptyCopy = document.getElementById("emptyCopy")
  refs.messageList = document.getElementById("messageList")
  refs.attachmentTray = document.getElementById("attachmentTray")
  refs.composerHint = document.getElementById("composerHint")
  refs.thinkingToggle = document.getElementById("thinkingToggle")
  refs.imagePickerButton = document.getElementById("imagePickerButton")
  refs.imageInput = document.getElementById("imageInput")
  refs.questionInput = document.getElementById("questionInput")
  refs.sendButton = document.getElementById("sendButton")
  refs.pdfDrawer = document.getElementById("pdfDrawer")
  refs.pdfDrawerBackdrop = document.getElementById("pdfDrawerBackdrop")
  refs.pdfDrawerTitle = document.getElementById("pdfDrawerTitle")
  refs.pdfDrawerMeta = document.getElementById("pdfDrawerMeta")
  refs.pdfDrawerClose = document.getElementById("pdfDrawerClose")
  refs.pdfDrawerResizeHandle = document.getElementById("pdfDrawerResizeHandle")
  refs.pdfFrameShell = document.getElementById("pdfFrameShell")
  refs.pdfPreviewFrame = document.getElementById("pdfPreviewFrame")
}

function bindEvents() {
  refs.brandButton.addEventListener("click", () => setSection("ask"))
  refs.navAskButton.addEventListener("click", () => setSection("ask"))
  refs.navDocumentsButton.addEventListener("click", () => setSection("documents"))
  refs.uploadButton.addEventListener("click", () => refs.fileInput.click())
  refs.docsUploadButton.addEventListener("click", () => refs.fileInput.click())
  refs.fileInput.addEventListener("change", async (event) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }
    await uploadDocument(file)
    refs.fileInput.value = ""
  })

  refs.documentSearchInput.addEventListener("input", (event) => {
    state.documentSearchQuery = String(event.target.value || "").trim().toLowerCase()
    renderDocuments()
  })
  refs.documentFilterBar.addEventListener("click", (event) => {
    const button = event.target.closest("[data-document-filter]")
    if (!button) {
      return
    }
    state.documentFilter = button.dataset.documentFilter || "all"
    renderDocuments()
  })
  refs.jumpToDocumentsButton.addEventListener("click", () => setSection("documents"))

  refs.globalChatButton.addEventListener("click", async () => {
    state.preferredMode = "global"
    state.activeConversationId = null
    state.messages = []
    state.currentSection = "ask"
    renderAll()
    setStatusLine("全局模式", "info")
  })

  refs.singleChatButton.addEventListener("click", async () => {
    state.preferredMode = "single"
    if (!state.activeDocumentId && state.documents[0]) {
      state.activeDocumentId = state.documents[0].id
    }
    if (!state.activeDocumentId) {
      setStatusLine("请先上传并选择一份文档，再开始单文件问答。", "warning")
      renderAll()
      return
    }
    state.activeConversationId = null
    state.messages = []
    state.currentSection = "ask"
    renderAll()
    setStatusLine(state.activeDocumentId ? `${documentLabel(currentDocument())} 单文件` : "单文件", "info")
  })

  refs.imagePickerButton.addEventListener("click", () => refs.imageInput.click())
  refs.imageInput.addEventListener("change", (event) => {
    const files = Array.from(event.target.files || [])
    if (!files.length) {
      return
    }

    const nextImages = files.map((file, index) => ({
      id: `${Date.now()}-${index}`,
      name: file.name,
      size: file.size,
    }))
    state.pendingImages = [...state.pendingImages, ...nextImages]
    refs.imageInput.value = ""
    renderAttachmentTray()
    setStatusLine("图片上传入口已预留。本版本先仅发送文本问题，图片不会上传到模型。", "info")
  })

  refs.questionInput.addEventListener("input", () => {
    autoResizeTextarea()
    syncComposerState()
  })

  refs.questionInput.addEventListener("keydown", async (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      await sendQuestion()
    }
  })

  refs.statusActionButton.addEventListener("click", async () => {
    const document = currentDocument()
    if (!document || state.isSubmitting || !isDocumentIndexFailed(document)) {
      return
    }
    await rebuildDocumentIndex(document.id)
  })

  refs.sendButton.addEventListener("click", async () => {
    await sendQuestion()
  })

  refs.pdfDrawerClose.addEventListener("click", closePdfPanel)
  refs.pdfDrawerBackdrop.addEventListener("click", closePdfPanel)
  refs.pdfDrawerResizeHandle.addEventListener("pointerdown", startPdfDrawerResize)

  refs.messageList.addEventListener(
    "scroll",
    () => {
      shouldAutoScroll = isMessageListNearBottom()
    },
    { passive: true },
  )

  refs.messageList.addEventListener("click", async (event) => {
    const thinkingSummary = event.target.closest(".thinking-block > summary")
    if (thinkingSummary) {
      event.preventDefault()

      const container = thinkingSummary.closest("[data-message-id]")
      const messageId = container?.dataset.messageId
      const message = state.messages.find((item) => String(item.id) === String(messageId))
      if (!message) {
        return
      }

      shouldAutoScroll = false
      message.thinking_expanded = !isThinkingExpanded(message)
      patchMessageElement(message)
      return
    }

    const citation = event.target.closest(".page-citation")
    if (citation) {
      const currentDoc = currentDocument()
      const page = Number(citation.dataset.page || 1)
      if (!currentDoc || !page) {
        return
      }

      await openPdfPanel(currentDoc, page)
      return
    }

    return
  })

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.pdfPanel.visible) {
      closePdfPanel()
    }
  })

  window.addEventListener("resize", applyPdfDrawerWidth)
}

function configureMarkdown() {
  if (window.marked?.setOptions) {
    window.marked.setOptions({
      gfm: true,
      breaks: true,
    })
  }

  if (window.mermaid?.initialize) {
    window.mermaid.initialize({ startOnLoad: false, theme: "neutral" })
  }
}

function citationLabel(page) {
  return `[p.${page}]`
}

function isShortPageReference(rawMatch) {
  return /^(?:\[\d+(?:\s*[–\-~至]\s*\d+)?\]|[Pp](?:age)?\.?\s*\d+(?:\s*[–\-~]\s*\d+)?)$/.test(
    String(rawMatch || "").trim(),
  )
}

function hasPageReference(text) {
  PAGE_REF_PATTERN.lastIndex = 0
  const matched = PAGE_REF_PATTERN.test(text)
  PAGE_REF_PATTERN.lastIndex = 0
  return matched
}

function injectPageCitations(html) {
  if (!html || !hasPageReference(html)) {
    return html
  }

  const template = document.createElement("template")
  template.innerHTML = html

  const walker = document.createTreeWalker(
    template.content,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        const parentElement = node.parentElement
        if (!parentElement) {
          return NodeFilter.FILTER_REJECT
        }
        if (!node.nodeValue?.trim()) {
          return NodeFilter.FILTER_REJECT
        }
        if (parentElement.closest("code, pre, a, button, .page-citation")) {
          return NodeFilter.FILTER_REJECT
        }
        return hasPageReference(node.nodeValue)
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT
      },
    },
  )

  const textNodes = []
  let currentNode = walker.nextNode()
  while (currentNode) {
    textNodes.push(currentNode)
    currentNode = walker.nextNode()
  }

  textNodes.forEach((node) => {
    const source = node.nodeValue || ""
    PAGE_REF_PATTERN.lastIndex = 0
    let cursor = 0
    let lastInsertedPage = null
    let lastInsertedEnd = -1
    let match = PAGE_REF_PATTERN.exec(source)
    if (!match) {
      return
    }

    const fragment = document.createDocumentFragment()
    while (match) {
      const [rawMatch] = match
      const startIndex = match.index
      const endIndex = startIndex + rawMatch.length
      const startPage = Number(match[1] || match[3] || match[5] || 1)
      const shortRef = isShortPageReference(rawMatch)
      const beforeChar = source[startIndex - 1] || ""
      const afterChar = source[endIndex] || ""
      const bracketPairs = {
        "[": "]",
        "(": ")",
        "（": "）",
        "【": "】",
      }
      const wrappedShortRef = shortRef && bracketPairs[beforeChar] === afterChar
      const replaceStart = wrappedShortRef ? startIndex - 1 : startIndex
      const replaceEnd = wrappedShortRef ? endIndex + 1 : endIndex
      const gapSinceLastCitation = source.slice(Math.max(cursor, lastInsertedEnd), replaceStart)
      const isAdjacentDuplicateShortRef =
        shortRef &&
        startPage === lastInsertedPage &&
        /^[\s[(（【]*$/.test(gapSinceLastCitation)

      if (isAdjacentDuplicateShortRef) {
        cursor = replaceEnd
        match = PAGE_REF_PATTERN.exec(source)
        continue
      }

      if (replaceStart > cursor) {
        fragment.append(document.createTextNode(source.slice(cursor, replaceStart)))
      }

      if (!shortRef) {
        fragment.append(document.createTextNode(rawMatch))
      }

      const citation = document.createElement("button")
      citation.type = "button"
      citation.className = "page-citation"
      citation.dataset.page = String(startPage)
      citation.setAttribute("aria-label", `查看第 ${startPage} 页原文`)
      citation.textContent = citationLabel(startPage)
      fragment.append(citation)

      lastInsertedPage = startPage
      lastInsertedEnd = replaceEnd
      cursor = replaceEnd
      match = PAGE_REF_PATTERN.exec(source)
    }

    if (cursor < source.length) {
      fragment.append(document.createTextNode(source.slice(cursor)))
    }

    node.parentNode.replaceChild(fragment, node)
  })

  return template.innerHTML
}

function buildPdfViewerUrl(doc, page) {
  const targetPage = Math.max(1, Number(page) || 1)
  const pdfUrl = doc?.pdf_url || `/api/documents/${doc?.id}/pdf`
  const absolutePdfUrl = new URL(pdfUrl, window.location.origin).href
  return `/static/pdfjs/web/viewer.html?file=${encodeURIComponent(absolutePdfUrl)}#page=${targetPage}`
}

function clampPdfDrawerWidth(width) {
  const maxWidth = Math.min(PDF_DRAWER_MAX_WIDTH, window.innerWidth - 32)
  return Math.min(Math.max(width, PDF_DRAWER_MIN_WIDTH), Math.max(PDF_DRAWER_MIN_WIDTH, maxWidth))
}

function applyPdfDrawerWidth() {
  refs.pdfDrawer.style.setProperty("--pdf-drawer-width", `${clampPdfDrawerWidth(state.pdfPanel.width)}px`)
}

function startPdfDrawerResize(event) {
  if (window.innerWidth <= 720) {
    return
  }

  event.preventDefault()
  const startX = event.clientX
  const startWidth = clampPdfDrawerWidth(state.pdfPanel.width)
  document.body.classList.add("is-drawer-resizing")

  const handlePointerMove = (moveEvent) => {
    state.pdfPanel.width = clampPdfDrawerWidth(startWidth + (startX - moveEvent.clientX))
    applyPdfDrawerWidth()
  }

  const stopResize = () => {
    document.body.classList.remove("is-drawer-resizing")
    window.removeEventListener("pointermove", handlePointerMove)
    window.removeEventListener("pointerup", stopResize)
  }

  window.addEventListener("pointermove", handlePointerMove)
  window.addEventListener("pointerup", stopResize, { once: true })
}

function mountPdfPreviewFrame(doc, page) {
  // 先清掉旧 frame，避免残留事件
  refs.pdfFrameShell.replaceChildren()
  refs.pdfPreviewFrame = null

  const frame = document.createElement("iframe")
  frame.className = "pdf-preview-frame"
  frame.title = "PDF 原文预览"

  // 先绑事件，再设 src，确保不错过 load 事件
  frame.addEventListener("load", handlePdfFrameLoad, { once: true })
  frame.addEventListener("error", handlePdfFrameError, { once: true })

  refs.pdfPreviewFrame = frame
  refs.pdfFrameShell.appendChild(frame)

  // src 最后赋值，触发加载
  frame.src = buildPdfViewerUrl(doc, page)
}

function closePdfPanel() {
  state.pdfPanel.visible = false
  state.pdfPanel.error = ""
  renderPdfDrawer()
}

function resetPdfPanel() {
  closePdfPanel()
  state.pdfPanel.documentId = null
  state.pdfPanel.filename = ""
  state.pdfPanel.pdfUrl = ""
  state.pdfPanel.page = 1
  refs.pdfFrameShell.replaceChildren()
  refs.pdfPreviewFrame = null
}

function renderPdfDrawer() {
  const panel = state.pdfPanel
  refs.pdfDrawer.classList.toggle("is-open", panel.visible)
  refs.pdfDrawerBackdrop.classList.toggle("is-visible", panel.visible)
  refs.pdfDrawer.setAttribute("aria-hidden", String(!panel.visible))

  refs.pdfDrawerTitle.textContent = panel.visible && panel.filename ? panel.filename : "未打开文档"

  if (!panel.visible) {
    refs.pdfDrawerMeta.textContent = "点击回答中的页码引用后，可在这里查看对应 PDF 原页。"
    refs.pdfFrameShell.hidden = true
    return
  }

  refs.pdfDrawerMeta.textContent = `${panel.filename} · 第 ${panel.page} 页`

  if (panel.error) {
    refs.pdfDrawerMeta.textContent = `加载失败：${panel.error}`
    refs.pdfFrameShell.hidden = true
    return
  }

  refs.pdfFrameShell.hidden = false
}

function openPdfPanel(doc, page) {
  const targetPage = Math.max(1, Number(page) || 1)
  state.pdfPanel.visible = true
  state.pdfPanel.documentId = doc.id
  state.pdfPanel.filename = doc.file_name
  state.pdfPanel.pdfUrl = doc.pdf_url || ""
  state.pdfPanel.page = targetPage
  state.pdfPanel.error = ""
  renderPdfDrawer()
  mountPdfPreviewFrame(doc, targetPage)
}

function handlePdfFrameLoad(event) {
  if (!state.pdfPanel.visible || event.currentTarget !== refs.pdfPreviewFrame) {
    return
  }
  state.pdfPanel.error = ""
  renderPdfDrawer()
}

function handlePdfFrameError(event) {
  if (!state.pdfPanel.visible || event.currentTarget !== refs.pdfPreviewFrame) {
    return
  }
  state.pdfPanel.error = "PDF 预览加载失败，请稍后重试。"
  renderPdfDrawer()
}

async function bootstrap() {
  try {
    applyPdfDrawerWidth()
    const data = await api("/api/bootstrap")
    state.documents = sortByUpdatedAt(data.documents || [])
    state.conversations = sortByUpdatedAt(data.conversations || [])
    renderSidebar()

    const restored = restoreStateFromHash()

    // 优先恢复 URL hash 中记录的会话
    if (restored.conversationId) {
      const conv = state.conversations.find((c) => c.id === restored.conversationId)
      if (conv) {
        await loadConversation(conv.id, false)
        const { message, kind } = buildDocumentIndexStatusLine(currentDocument())
        setStatusLine(message, kind)
        if (restored.section === "documents") {
          setSection("documents")
        }
        return
      }
    }

    // 其次恢复最后活跃的会话
    if (state.conversations[0]) {
      await loadConversation(state.conversations[0].id, false)
      const { message, kind } = buildDocumentIndexStatusLine(currentDocument())
      setStatusLine(message, kind)
      if (restored.section === "documents") {
        setSection("documents")
      }
      return
    }

    if (restored.documentId) {
      const doc = state.documents.find((d) => d.id === restored.documentId)
      if (doc) {
        state.activeDocumentId = doc.id
        renderWorkspace()
        const { message, kind } = buildDocumentIndexStatusLine(doc)
        setStatusLine(message, kind)
        if (restored.section === "documents") {
          setSection("documents")
        }
        return
      }
    }

    if (restored.section === "documents") {
      setSection("documents")
    }

    if (state.documents[0]) {
      renderWorkspace()
      setStatusLine("可提问", "info")
      return
    }

    renderWorkspace()
    setStatusLine("等待文档", "info")
  } catch (error) {
    console.error(error)
    renderWorkspace()
    setStatusLine(error.message || "初始化失败，请检查服务是否已启动。", "error")
  }
}

async function api(url, options = {}) {
  const response = await fetch(url, options)
  const contentType = response.headers.get("content-type") || ""
  const payload = contentType.includes("application/json")
    ? await response.json()
    : { detail: await response.text() }

  if (!response.ok) {
    throw new Error(payload.detail || "请求失败")
  }

  return payload
}

function hasPendingAssistantMessage() {
  return state.messages.some((message) => message.role === "assistant" && message.pending)
}

function maybeSetOcrStatusLine(message, kind = "info") {
  if (hasPendingAssistantMessage()) {
    return
  }
  setStatusLine(message, kind)
}

function sortByUpdatedAt(items) {
  return [...items].sort((a, b) => {
    const timeA = new Date(a.updated_at || a.created_at || 0).getTime()
    const timeB = new Date(b.updated_at || b.created_at || 0).getTime()
    return timeB - timeA
  })
}

function upsertById(items, item) {
  if (!item) {
    return [...items]
  }

  const next = [...items]
  const index = next.findIndex((candidate) => candidate.id === item.id)
  if (index >= 0) {
    next[index] = item
  } else {
    next.unshift(item)
  }
  return next
}

function currentDocument() {
  return state.documents.find((document) => document.id === state.activeDocumentId) || null
}

function currentConversation() {
  return state.conversations.find((conversation) => conversation.id === state.activeConversationId) || null
}

function currentMode() {
  const conversation = currentConversation()
  if (conversation) {
    return conversation.document_id ? "single" : "global"
  }
  return state.preferredMode
}

function setSection(section) {
  state.currentSection = section === "documents" ? "documents" : "ask"
  renderAll()
  persistStateToHash()
}

function hasAnyQueryableDocument() {
  return state.documents.some((document) => document.ocr_status === "done")
}

function documentOcrStatus(document) {
  return document?.ocr_status || "pending"
}

function documentOcrProgress(document) {
  const value = Number(document?.ocr_progress ?? 0)
  if (!Number.isFinite(value)) {
    return 0
  }
  return Math.max(0, Math.min(100, Math.round(value)))
}

function documentOcrDetail(document) {
  return String(document?.ocr_detail || "").trim()
}

function isDocumentIndexReady(document) {
  return documentOcrStatus(document) === "done"
}

function isDocumentIndexFailed(document) {
  return documentOcrStatus(document) === "failed"
}

function isDocumentIndexBuilding(document) {
  const status = documentOcrStatus(document)
  return status === "pending" || status === "processing"
}

function buildDocumentIndexLabel(document) {
  if (!document) {
    return ""
  }

  if (isDocumentIndexReady(document)) {
    return "就绪"
  }

  if (isDocumentIndexFailed(document)) {
    return "失败"
  }

  return `处理中 ${documentOcrProgress(document)}%`
}

function buildDocumentNavMeta(document) {
  if (!document) {
    return ""
  }
  return `${document.page_count} 页 · ${buildDocumentIndexLabel(document)}`
}

function filteredDocuments() {
  return state.documents.filter((document) => {
    const matchesSearch = !state.documentSearchQuery || [
      document.file_name,
      document.display_name,
      ...(document.title_aliases || []),
      ...(document.keywords || []),
    ]
      .join(" ")
      .toLowerCase()
      .includes(state.documentSearchQuery)

    if (!matchesSearch) {
      return false
    }

    if (state.documentFilter === "ready") {
      return isDocumentIndexReady(document)
    }
    if (state.documentFilter === "building") {
      return isDocumentIndexBuilding(document)
    }
    if (state.documentFilter === "failed") {
      return isDocumentIndexFailed(document)
    }
    return true
  })
}

function buildDocumentIndexStatusLine(document) {
  if (!document) {
    return {
      kind: "info",
      message: hasAnyQueryableDocument()
        ? "可提问"
        : "等待文档",
    }
  }

  const detail = documentOcrDetail(document)
  if (isDocumentIndexReady(document)) {
    return {
      kind: "success",
      message: detail || `${documentLabel(document)} 就绪`,
    }
  }

  if (isDocumentIndexFailed(document)) {
    return {
      kind: "warning",
      message: detail || `${documentLabel(document)} 失败`,
    }
  }

  const progress = documentOcrProgress(document)
  return {
    kind: "info",
    message: detail
      ? `${detail} ${progress}%`
      : `${documentLabel(document)} 处理中 ${progress}%`,
  }
}

function documentLabel(document) {
  return `${document.display_name} v${document.version_index}`
}

function latestConversationForDocument(documentId) {
  const matches = state.conversations.filter((conversation) => conversation.document_id === documentId)
  return sortByUpdatedAt(matches)[0] || null
}

function setStatusLine(message, kind = "info") {
  const normalized = String(message || "").trim()
  refs.statusText.textContent = normalized
  refs.statusBanner.className = `status-line status-line--${kind}${normalized ? "" : " is-dot-only"}`
  syncStatusActionButton()
  syncSidebarStatus()
}

function syncSidebarStatus() {
  const readyCount = state.documents.filter((document) => isDocumentIndexReady(document)).length
  if (!state.documents.length) {
    refs.sidebarStatusMeta.textContent = "0/0 就绪"
    return
  }
  refs.sidebarStatusMeta.textContent = `${readyCount}/${state.documents.length} 就绪`
}

function syncStatusActionButton() {
  const document = currentDocument()
  const canRebuild = Boolean(document && isDocumentIndexFailed(document) && !state.isSubmitting)
  refs.statusActionButton.hidden = !canRebuild
  refs.statusActionButton.disabled = !canRebuild
}

async function uploadDocument(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setStatusLine("仅支持上传 PDF 文件。", "warning")
    return
  }

  state.isSubmitting = true
  syncComposerState()
  setStatusLine(`正在处理 ${file.name}，首次上传会先完成页面缓存。`, "info")

  try {
    const formData = new FormData()
    formData.append("file", file)

    const result = await api("/api/documents", {
      method: "POST",
      body: formData,
    })

    state.documents = sortByUpdatedAt(upsertById(state.documents, result.document))
    state.conversations = sortByUpdatedAt(upsertById(state.conversations, result.conversation))
    state.currentSection = "ask"
    await loadConversation(result.conversation.id, false)

    const { message, kind } = buildDocumentIndexStatusLine(currentDocument() || result.document)
    setStatusLine(message, kind)

    void pollOcrStatus(result.document.id)
  } catch (error) {
    console.error(error)
    setStatusLine(error.message || "上传失败。", "error")
  } finally {
    state.isSubmitting = false
    syncComposerState()
  }
}

function patchDocumentRow(document) {
  if (!document) {
    return
  }

  const card = refs.documentLibraryGrid?.querySelector(`[data-document-card-id="${document.id}"]`)
  if (!card) {
    return
  }

  card.classList.toggle("is-active", document.id === state.activeDocumentId)
  const pages = card.querySelector(".document-row__pages")
  if (pages) {
    pages.textContent = `${document.page_count} 页`
  }

  // 更新状态徽章（OCR 状态变化时自动同步）
  const badgeEl = card.querySelector(".document-status-badge")
  if (badgeEl) {
    const badge = isDocumentIndexReady(document)
      ? "就绪"
      : isDocumentIndexFailed(document)
        ? "失败"
        : "处理中"
    const badgeClass = isDocumentIndexReady(document)
      ? "is-ready"
      : isDocumentIndexFailed(document)
        ? "is-failed"
        : ""
    badgeEl.textContent = badge
    badgeEl.className = `document-status-badge ${badgeClass}`
  }
}

function syncWorkspaceChrome() {
  const currentDoc = currentDocument()
  const conversation = currentConversation()
  const mode = currentMode()
  const isGlobalConversation = mode === "global"
  const hasMessages = state.messages.length > 0
  const readyToChat = hasAnyQueryableDocument()
  const indexProgress = documentOcrProgress(currentDoc)
  const indexDetail = documentOcrDetail(currentDoc)
  const cleanTitle = formatConversationTitle(conversation?.title || "")

  refs.workspace.dataset.section = state.currentSection
  refs.workspace.dataset.mode = state.currentSection === "ask" && hasMessages ? "chat" : "empty"
  refs.askView.hidden = state.currentSection !== "ask"
  refs.documentsView.hidden = state.currentSection !== "documents"

  refs.navAskButton.classList.toggle("is-active", state.currentSection === "ask")
  refs.navDocumentsButton.classList.toggle("is-active", state.currentSection === "documents")
  refs.globalChatButton.classList.toggle("is-active", mode === "global")
  refs.singleChatButton.classList.toggle("is-active", mode === "single")

  if (state.currentSection === "documents") {
    refs.headerBreadcrumb.textContent = "文档"
    refs.conversationTitle.textContent = "文档"
    refs.headerMeta.textContent = `${state.documents.length} 个文档`
    refs.currentDocumentPill.textContent = currentDoc ? `预览 ${documentLabel(currentDoc)}` : "文档"
  } else {
    refs.headerBreadcrumb.textContent = buildBreadcrumb(isGlobalConversation ? null : currentDoc, conversation)
    refs.conversationTitle.textContent = cleanTitle || (isGlobalConversation ? "问答" : currentDoc ? documentLabel(currentDoc) : "问答")
    refs.headerMeta.textContent = buildHeaderMeta(isGlobalConversation ? null : currentDoc, conversation, hasMessages)
    refs.currentDocumentPill.textContent = isGlobalConversation ? "全局模式" : "单文件"
  }

  if (isGlobalConversation) {
    refs.composerHint.textContent = ""
    refs.askModeTag.textContent = "全局模式"
    refs.askScopeText.textContent = currentDoc ? `预览 ${documentLabel(currentDoc)}` : ""
  } else if (!currentDoc) {
    refs.composerHint.textContent = ""
    refs.askModeTag.textContent = "单文件"
    refs.askScopeText.textContent = ""
  } else if (readyToChat) {
    refs.composerHint.textContent = ""
    refs.askModeTag.textContent = "单文件"
    refs.askScopeText.textContent = documentLabel(currentDoc)
  } else if (isDocumentIndexFailed(currentDoc)) {
    refs.composerHint.textContent = ""
    refs.askModeTag.textContent = "单文件"
    refs.askScopeText.textContent = `${documentLabel(currentDoc)} 失败`
  } else {
    refs.composerHint.textContent = ""
    refs.askModeTag.textContent = "单文件"
    refs.askScopeText.textContent = `${documentLabel(currentDoc)} ${indexProgress}%`
  }

  if (isGlobalConversation || !currentDoc) {
    refs.emptyTitle.textContent = "开始问答"
    refs.emptyCopy.textContent = isGlobalConversation ? "输入问题" : "先选择文档"
  } else if (!hasMessages) {
    if (readyToChat) {
      refs.emptyTitle.textContent = documentLabel(currentDoc)
      refs.emptyCopy.textContent = "输入问题"
    } else if (isDocumentIndexFailed(currentDoc)) {
      refs.emptyTitle.textContent = documentLabel(currentDoc)
      refs.emptyCopy.textContent = indexDetail || "索引失败"
    } else {
      refs.emptyTitle.textContent = documentLabel(currentDoc)
      refs.emptyCopy.textContent = indexDetail || `处理中 ${indexProgress}%`
    }
  }

  renderAttachmentTray()
  syncComposerState()
  syncPendingWaitTicker()
  syncStatusActionButton()
}

function refreshDocumentIndexUi(documentId, options = {}) {
  const { refreshWorkspace = false } = options
  const document = state.documents.find((item) => item.id === documentId)
  patchDocumentRow(document)

  if (documentId !== state.activeDocumentId) {
    return
  }

  const { message, kind } = buildDocumentIndexStatusLine(document)
  maybeSetOcrStatusLine(message, kind)
  syncStatusActionButton()

  // 后台轮询时也要更新空面板和状态栏中的进度文字，避免调用 syncWorkspaceChrome 导致闪烁
  const indexProgress = documentOcrProgress(document)
  const indexDetail = documentOcrDetail(document)
  if (refs.workspace.dataset.mode === "empty") {
    if (isDocumentIndexFailed(document)) {
      refs.emptyCopy.textContent = indexDetail || "索引失败"
    } else if (!isDocumentIndexReady(document)) {
      refs.emptyCopy.textContent = indexDetail || `处理中 ${indexProgress}%`
    }
  }
  if (refs.askScopeText && !isDocumentIndexReady(document) && !isDocumentIndexFailed(document)) {
    refs.askScopeText.textContent = `${documentLabel(document)} ${indexProgress}%`
  }

  // refreshWorkspace 仅由用户主动操作（上传、重建、选择文档）触发
  if (refreshWorkspace) {
    syncWorkspaceChrome()
  }
}

async function pollOcrStatus(documentId) {
  if (activeOcrPolls.has(documentId)) {
    return
  }

  activeOcrPolls.add(documentId)
  const maxWait = 10 * 60 * 1000
  const start = Date.now()

  try {
    while (Date.now() - start < maxWait) {
      const result = await api(`/api/documents/${documentId}/ocr-status`)
      const status = result.ocr_status || "pending"
      const document = state.documents.find((item) => item.id === documentId)
      const previousStatus = document?.ocr_status || "pending"
      const nextProgress = Number(result.ocr_progress || 0)
      const nextDetail = result.ocr_detail || ""
      let changed = true

      if (document) {
        changed =
          document.ocr_status !== status ||
          Number(document.ocr_progress || 0) !== nextProgress ||
          String(document.ocr_detail || "") !== nextDetail

        document.ocr_status = status
        document.ocr_progress = nextProgress
        document.ocr_detail = nextDetail
      }

      if (changed) {
        refreshDocumentIndexUi(documentId)
      }

      if (status === "done") {
        return
      }

      if (status === "failed") {
        return
      }

      await new Promise((resolve) => window.setTimeout(resolve, 5000))
    }
  } catch (error) {
    console.error(error)
  } finally {
    activeOcrPolls.delete(documentId)
  }
}

async function handleDocumentSelect(documentId) {
  resetPdfPanel()
  state.activeDocumentId = documentId
  persistStateToHash()
  if (currentMode() === "global") {
    renderAll()
    const { message, kind } = buildDocumentIndexStatusLine(currentDocument())
    setStatusLine(
      currentDocument()
        ? `${documentLabel(currentDocument())} 预览`
        : message,
      currentDocument() ? "info" : kind,
    )
    return
  }

  state.activeConversationId = null
  state.messages = []
  state.currentSection = "ask"
  renderAll()
  persistStateToHash()

  const conversation = latestConversationForDocument(documentId)
  if (conversation) {
    await loadConversation(conversation.id, false)
    const { message, kind } = buildDocumentIndexStatusLine(currentDocument())
    setStatusLine(message, kind)
    return
  }

  await createConversation(documentId)
}

async function createConversation(documentId) {
  state.isSubmitting = true
  syncComposerState()

  try {
    const result = await api("/api/conversations", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ document_id: documentId }),
    })

    if (result.document) {
      state.documents = sortByUpdatedAt(upsertById(state.documents, result.document))
    }
    state.conversations = sortByUpdatedAt(upsertById(state.conversations, result.conversation))
    state.activeDocumentId = result.document?.id || null
    state.activeConversationId = result.conversation.id
    state.preferredMode = result.conversation?.document_id ? "single" : "global"
    state.currentSection = "ask"
    resetPdfPanel()
    state.messages = []
    renderAll()
    const { message, kind } = buildDocumentIndexStatusLine(result.document || currentDocument())
    setStatusLine(message, kind)
    if (result.document && !isDocumentIndexReady(result.document)) {
      void pollOcrStatus(result.document.id)
    }
  } catch (error) {
    console.error(error)
    setStatusLine(error.message || "新建会话失败。", "error")
  } finally {
    state.isSubmitting = false
    syncComposerState()
  }
}

async function deleteDocument(documentId) {
  const document = state.documents.find((item) => item.id === documentId)
  if (!document) {
    return
  }

  const confirmed = window.confirm(
    `确认删除 ${document.file_name} 吗？\n该文档绑定的会话和消息也会一并删除。`,
  )
  if (!confirmed) {
    return
  }

  state.isSubmitting = true
  syncComposerState()
  setStatusLine(`正在删除 ${documentLabel(document)}...`, "warning")

  try {
    const result = await api(`/api/documents/${documentId}`, {
      method: "DELETE",
    })

    const deletedConversationIds = new Set(result.deleted_conversation_ids || [])
    state.documents = state.documents.filter((item) => item.id !== documentId)
    state.conversations = state.conversations.filter(
      (conversation) => !deletedConversationIds.has(conversation.id),
    )

    const deletedActiveDocument = state.activeDocumentId === documentId
    const deletedActiveConversation = deletedConversationIds.has(state.activeConversationId)

    resetPdfPanel()

    if (deletedActiveDocument || deletedActiveConversation) {
      state.activeDocumentId = null
      state.activeConversationId = null
      state.messages = []
      state.currentSection = "ask"

      if (state.conversations[0]) {
        await loadConversation(state.conversations[0].id, false)
      } else if (state.documents[0]) {
        await handleDocumentSelect(state.documents[0].id)
      } else {
        renderAll()
        persistStateToHash()
      }
    } else {
      renderSidebar()
      renderWorkspace()
    }

    setStatusLine(`${documentLabel(document)} 已删除。`, "success")
  } catch (error) {
    console.error(error)
    setStatusLine(error.message || "删除文档失败。", "error")
  } finally {
    state.isSubmitting = false
    syncComposerState()
  }
}

async function deleteConversation(conversationId) {
  const conversation = state.conversations.find((item) => item.id === conversationId)
  if (!conversation) {
    return
  }

  const confirmed = window.confirm(
    `确认删除会话「${conversation.title}」吗？\n该会话下的消息会一并删除。`,
  )
  if (!confirmed) {
    return
  }

  state.isSubmitting = true
  syncComposerState()
  setStatusLine(`正在删除会话「${conversation.title}」...`, "warning")

  try {
    const result = await api(`/api/conversations/${conversationId}`, {
      method: "DELETE",
    })

    state.conversations = state.conversations.filter((item) => item.id !== conversationId)
    const deletedActiveConversation = state.activeConversationId === conversationId

    if (deletedActiveConversation) {
      resetPdfPanel()
      state.activeConversationId = null
      state.messages = []

      const fallbackDocumentId = Number(result.document_id || conversation.document_id || 0) || null
      const nextSameDocumentConversation = fallbackDocumentId
        ? latestConversationForDocument(fallbackDocumentId)
        : null

      if (nextSameDocumentConversation) {
        await loadConversation(nextSameDocumentConversation.id, false)
      } else if (state.conversations[0]) {
        await loadConversation(state.conversations[0].id, false)
      } else {
        state.activeDocumentId = fallbackDocumentId || null
        state.currentSection = "ask"
        renderAll()
        persistStateToHash()
      }
    } else {
      renderSidebar()
    }

    setStatusLine(`会话「${conversation.title}」已删除。`, "success")
  } catch (error) {
    console.error(error)
    setStatusLine(error.message || "删除会话失败。", "error")
  } finally {
    state.isSubmitting = false
    syncComposerState()
  }
}

async function rebuildDocumentIndex(documentId) {
  const document = state.documents.find((item) => item.id === documentId)
  if (!document) {
    return
  }

  state.isSubmitting = true
  syncComposerState()
  setStatusLine(`正在重新提交 ${documentLabel(document)} 的索引任务...`, "info")

  try {
    const result = await api(`/api/documents/${documentId}/rebuild-index`, {
      method: "POST",
    })

    document.ocr_status = result.ocr_status || "processing"
    document.ocr_progress = Number(result.ocr_progress || 0)
    document.ocr_detail = result.detail || "已提交索引重建任务。"

    refreshDocumentIndexUi(documentId, { refreshWorkspace: true })
    void pollOcrStatus(documentId)
  } catch (error) {
    console.error(error)
    setStatusLine(error.message || "重建索引失败。", "error")
  } finally {
    state.isSubmitting = false
    syncComposerState()
    syncStatusActionButton()
  }
}

async function loadConversation(conversationId, announce = true) {
  state.isSubmitting = true
  syncComposerState()

  try {
    const result = await api(`/api/conversations/${conversationId}`)
    state.conversations = sortByUpdatedAt(upsertById(state.conversations, result.conversation))
    if (result.document) {
      state.documents = sortByUpdatedAt(upsertById(state.documents, result.document))
    }
    ;(result.routed_documents || []).forEach((document) => {
      state.documents = sortByUpdatedAt(upsertById(state.documents, document))
    })
    state.activeConversationId = result.conversation.id
    state.activeDocumentId = result.conversation?.document_id || result.document?.id || null
    state.preferredMode = result.conversation?.document_id ? "single" : "global"
    state.currentSection = "ask"
    resetPdfPanel()
    persistStateToHash()
    state.messages = (result.messages || []).map((message) => ({
      ...message,
      metadata: message.metadata || {},
      thinking_text: "",
      thinking_expanded: false,
    }))
    shouldAutoScroll = true
    renderAll()
    scrollMessagesToBottom(true)

    if (result.document && !isDocumentIndexReady(result.document)) {
      void pollOcrStatus(result.document.id)
    }

    if (announce) {
      const { message, kind } = buildDocumentIndexStatusLine(result.document || currentDocument())
      setStatusLine(message, kind)
    }
  } catch (error) {
    console.error(error)
    setStatusLine(error.message || "加载会话失败。", "error")
  } finally {
    state.isSubmitting = false
    syncComposerState()
  }
}

async function sendQuestion() {
  const question = refs.questionInput.value.trim()
  if (!state.activeConversationId) {
    await createConversation(currentMode() === "single" ? state.activeDocumentId : null)
  }

  if (!state.activeConversationId) {
    setStatusLine("当前没有可用会话，请稍后重试。", "warning")
    return
  }

  if (currentMode() === "single" && !state.activeDocumentId) {
    setStatusLine("单文件问答模式下，请先选择一份文档。", "warning")
    return
  }

  if (!hasAnyQueryableDocument()) {
    setStatusLine("至少需要一份已完成 OCR 的文档后才能提问。", "warning")
    return
  }

  if (!question || state.isSubmitting) {
    return
  }

  if (state.pendingImages.length) {
    setStatusLine("图片入口已预留，当前版本仍只发送文本问题。", "warning")
  }

  const tempUserId = `temp-user-${Date.now()}`
  const tempAssistantId = `temp-assistant-${Date.now()}`
  const tempUser = {
    id: tempUserId,
    role: "user",
    content: question,
    created_at: new Date().toISOString(),
  }
  const tempAssistant = {
    id: tempAssistantId,
    role: "assistant",
    content: "",
    thinking_text: "",
    thinking_expanded: true,
    pending_started_at: Date.now(),
    progress_stage: "正在处理用户问题",
    progress_detail: "请耐心等待，正在处理你的问题。",
    created_at: new Date().toISOString(),
    pending: true,
  }

  refs.questionInput.value = ""
  autoResizeTextarea()

  state.isSubmitting = true
  state.messages = [...state.messages, tempUser, tempAssistant]
  shouldAutoScroll = true
  renderWorkspace()
  scrollMessagesToBottom(true)
  setStatusLine("正在处理你的问题。", "info")

  try {
    await streamConversation(question, tempUserId, tempAssistantId)
  } catch (error) {
    console.error(error)
    state.messages = state.messages.filter(
      (message) => message.id !== tempUserId && message.id !== tempAssistantId,
    )
    refs.questionInput.value = question
    autoResizeTextarea()
    renderWorkspace()
    setStatusLine(error.message || "发送失败。", "error")
  } finally {
    state.isSubmitting = false
    syncComposerState()
  }
}

async function streamConversation(question, tempUserId, tempAssistantId) {
  const response = await fetch(`/api/conversations/${state.activeConversationId}/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      enable_thinking: refs.thinkingToggle.checked,
    }),
  })

  if (!response.ok) {
    let detail = "请求失败"
    try {
      const payload = await response.json()
      detail = payload.detail || detail
    } catch (error) {
      console.error(error)
    }
    throw new Error(detail)
  }

  if (!response.body) {
    throw new Error("当前浏览器不支持流式响应读取。")
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let receivedDone = false

  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })

    let separatorIndex = buffer.indexOf("\n")
    while (separatorIndex >= 0) {
      const rawLine = buffer.slice(0, separatorIndex).trim()
      buffer = buffer.slice(separatorIndex + 1)

      if (rawLine) {
        const event = JSON.parse(rawLine)
        const finished = handleStreamEvent(event, tempUserId, tempAssistantId)
        if (finished) {
          receivedDone = true
        }
      }

      separatorIndex = buffer.indexOf("\n")
    }

    if (done) {
      break
    }
  }

  if (buffer.trim()) {
    const event = JSON.parse(buffer.trim())
    const finished = handleStreamEvent(event, tempUserId, tempAssistantId)
    if (finished) {
      receivedDone = true
    }
  }

  if (!receivedDone) {
    throw new Error("连接已结束，但没有收到完整回答。")
  }
}

function handleStreamEvent(event, tempUserId, tempAssistantId) {
  const tempAssistant = state.messages.find((message) => message.id === tempAssistantId)

  if (event.type === "progress") {
    if (tempAssistant) {
      const stage = event.stage || "正在处理中"
      const detail = event.detail || ""
      tempAssistant.progress_stage = stage
      tempAssistant.progress_detail = detail
    }

    setStatusLine(buildPendingWaitMessage(tempAssistant), "info")
    return false
  }

  if (event.type === "started") {
    const document = event.conversation?.document_id ? (event.document || currentDocument()) : null
    if (tempAssistant) {
      const stage = "模型已开始响应"
      const detail = "已建立响应流，正在等待思考过程或回答正文。"
      tempAssistant.progress_stage = stage
      tempAssistant.progress_detail = detail
    }

    if (document) {
      setStatusLine(`请耐心等待，正在基于 ${documentLabel(document)} 分析文档。`, "info")
    } else {
      setStatusLine("请耐心等待，模型正在分析文档。", "info")
    }
    return false
  }

  if (event.type === "thinking_delta") {
    if (tempAssistant) {
      tempAssistant.thinking_text += event.delta || ""
      if (isThinkingExpanded(tempAssistant)) {
        scheduleWorkspaceRender()
      }
    }
    return false
  }

  if (event.type === "answer_delta") {
    if (tempAssistant) {
      tempAssistant.content += event.delta || ""
      scheduleWorkspaceRender()
    }
    return false
  }

  if (event.type === "error") {
    throw new Error(event.detail || "模型请求失败。")
  }

  if (event.type === "done") {
    state.messages = state.messages.filter(
      (message) => message.id !== tempUserId && message.id !== tempAssistantId,
    )

    const finalThinkingText = (event.thinking_text || tempAssistant?.thinking_text || "").trim()

    const assistantMessage = {
      ...event.assistant_message,
      metadata: event.assistant_message?.metadata || {},
      thinking_text: finalThinkingText,
      thinking_expanded: Boolean(tempAssistant?.thinking_expanded) && Boolean(finalThinkingText),
    }

    state.messages.push(event.user_message, assistantMessage)
    if (event.document) {
      state.documents = sortByUpdatedAt(upsertById(state.documents, event.document))
    }
    ;(event.routed_documents || []).forEach((document) => {
      state.documents = sortByUpdatedAt(upsertById(state.documents, document))
    })
    state.conversations = sortByUpdatedAt(upsertById(state.conversations, event.conversation))
    state.activeDocumentId = event.conversation?.document_id || event.document?.id || null
    state.activeConversationId = event.conversation.id
    state.preferredMode = event.conversation?.document_id ? "single" : "global"
    state.currentSection = "ask"
    shouldAutoScroll = true
    persistStateToHash()
    renderAll()
    scrollMessagesToBottom(true)
    setStatusLine("", "success")
    return true
  }

  return false
}

async function enterSingleDocumentChat(documentId) {
  state.preferredMode = "single"
  state.activeDocumentId = documentId
  state.currentSection = "ask"
  const existingConversation = latestConversationForDocument(documentId)
  if (existingConversation) {
    await loadConversation(existingConversation.id, false)
    const { message, kind } = buildDocumentIndexStatusLine(currentDocument())
    setStatusLine(message, kind)
    return
  }
  await createConversation(documentId)
}

async function previewDocument(documentId, page = 1) {
  const document = state.documents.find((item) => item.id === documentId)
  if (!document) {
    return
  }
  state.activeDocumentId = document.id
  renderAll()
  if (document.pdf_url) {
    await openPdfPanel(document, page)
  } else {
    setStatusLine(`${documentLabel(document)} 已设为当前预览文档。`, "info")
  }
}

function scheduleWorkspaceRender() {
  if (workspaceRenderQueued) return
  workspaceRenderQueued = true
  window.requestAnimationFrame(() => {
    workspaceRenderQueued = false
    patchStreamingMessage()     // ← 只动这一条，不动整个列表
    scrollMessagesToBottom()
  })
}

function patchMessageElement(message) {
  if (!message) {
    return
  }

  const existing = refs.messageList.querySelector(`[data-message-id="${message.id}"]`)

  if (existing) {
    if (message.role === "assistant") {
      existing.className = `message message--assistant ${message.pending ? "is-pending" : ""}`.trim()
      const flow = existing.querySelector(".assistant-flow")
      if (flow) {
        flow.innerHTML = `${renderThinkingBlock(message)}${renderAssistantBody(message)}`
      }
      // 思考框内容更新后自动滚动到底部
      if (message.pending && isThinkingExpanded(message)) {
        const thinkingBody = existing.querySelector(".thinking-body")
        if (thinkingBody) {
          thinkingBody.scrollTop = thinkingBody.scrollHeight
        }
      }
      return
    }

    if (message.role === "user") {
      existing.className = `message message--user ${message.pending ? "is-pending" : ""}`.trim()
      const bubble = existing.querySelector(".user-bubble")
      if (bubble) {
        bubble.innerHTML = renderPlainText(message.content)
      }
      return
    }
  }

  const tpl = document.createElement("template")
  tpl.innerHTML = renderMessage(message)
  refs.messageList.appendChild(tpl.content.firstElementChild)
}

function patchStreamingMessage() {
  const pending = state.messages.find((m) => m.pending)
  if (!pending) {
    renderWorkspace()
    return
  }

  // 确保非 pending 的消息已经渲染到 DOM（首次进入 chat 模式）
  if (refs.workspace.dataset.mode !== "chat") {
    renderWorkspace()
    return
  }
  patchMessageElement(pending)
}

function renderAll() {
  renderSidebar()
  renderWorkspace()
}

function renderSidebar() {
  refs.navAskButton.classList.toggle("is-active", state.currentSection === "ask")
  refs.navDocumentsButton.classList.toggle("is-active", state.currentSection === "documents")
  syncSidebarStatus()
  renderDocuments()
  renderConversations()
}

function renderDocuments() {
  const filtered = filteredDocuments()

  refs.documentMeta.textContent = `${filtered.length || 0} 个文档`

  refs.documentFilterBar.querySelectorAll("[data-document-filter]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.documentFilter === state.documentFilter)
  })

  if (!state.documents.length) {
    refs.documentLibraryGrid.innerHTML = '<div class="library-empty">暂无文档</div>'
    return
  }

  if (!filtered.length) {
    refs.documentLibraryGrid.innerHTML = '<div class="library-empty">无匹配结果</div>'
    return
  }

  refs.documentLibraryGrid.innerHTML = filtered
    .map((document) => {
      const activeClass = document.id === state.activeDocumentId ? "is-active" : ""
      const badge = isDocumentIndexReady(document)
        ? "就绪"
        : isDocumentIndexFailed(document)
          ? "失败"
          : "处理中"
      const badgeClass = isDocumentIndexReady(document)
        ? "is-ready"
        : isDocumentIndexFailed(document)
          ? "is-failed"
          : ""
      const aliases = (document.title_aliases || []).slice(0, 3).join(" · ")
      const keywords = (document.keywords || []).slice(0, 4).join(" · ")
      const detail = aliases || keywords || "—"
      return `
        <article class="document-row ${activeClass}" data-document-card-id="${document.id}">
          <div class="document-row__main">
            <div class="document-row__title">
              <span class="document-row__name">${escapeHtml(document.file_name)}</span>
              <span class="version-badge">v${document.version_index}</span>
            </div>
            <div class="document-row__detail">${escapeHtml(detail)}</div>
          </div>
          <div class="document-status-badge ${badgeClass}">${badge}</div>
          <div class="document-row__pages">${document.page_count} 页</div>
          <div class="document-row__actions">
            <button class="doc-action doc-action--primary" type="button" data-document-chat="${document.id}">问答</button>
            <button class="doc-action" type="button" data-document-open="${document.id}">查看原文</button>
            ${badge === "失败" ? `<button class="doc-action doc-action--warn" type="button" data-document-rebuild="${document.id}">重建索引</button>` : ""}
            <button class="doc-action doc-action--icon" type="button" data-document-delete="${document.id}" aria-label="删除">
              <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
                <path
                  d="M6.5 7.5v7m3.5-7v7m3.5-7v7M4 5.5h12m-8.5 0 .7-1.4A1 1 0 0 1 9.1 3.5h1.8a1 1 0 0 1 .9.6l.7 1.4m-7.2 0 .4 9.3a1.2 1.2 0 0 0 1.2 1.2h6.2a1.2 1.2 0 0 0 1.2-1.2l.4-9.3"
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="1.5"
                />
              </svg>
            </button>
          </div>
        </article>
      `
    })
    .join("")

  refs.documentLibraryGrid.querySelectorAll("[data-document-chat]").forEach((element) => {
    element.addEventListener("click", async () => {
      await enterSingleDocumentChat(Number(element.dataset.documentChat))
    })
  })

  refs.documentLibraryGrid.querySelectorAll("[data-document-open]").forEach((element) => {
    element.addEventListener("click", async () => {
      await previewDocument(Number(element.dataset.documentOpen), 1)
    })
  })

  refs.documentLibraryGrid.querySelectorAll("[data-document-delete]").forEach((element) => {
    const triggerDelete = async (event) => {
      event.preventDefault()
      event.stopPropagation()
      if (state.isSubmitting) {
        return
      }
      await deleteDocument(Number(element.dataset.documentDelete))
    }

    element.addEventListener("click", triggerDelete)
    element.addEventListener("keydown", async (event) => {
      if (event.key === "Enter" || event.key === " ") {
        await triggerDelete(event)
      }
    })
  })

  refs.documentLibraryGrid.querySelectorAll("[data-document-rebuild]").forEach((element) => {
    element.addEventListener("click", async () => {
      if (state.isSubmitting) {
        return
      }
      await rebuildDocumentIndex(Number(element.dataset.documentRebuild))
    })
  })
}

function renderConversations() {
  const filtered = state.conversations

  if (!state.conversations.length) {
    refs.conversationMeta.textContent = "0 条记录"
    refs.conversationLibraryList.innerHTML = '<div class="library-empty">暂无会话</div>'
    return
  }

  refs.conversationMeta.textContent = `${filtered.length} 条记录`

  refs.conversationLibraryList.innerHTML = filtered
    .map((conversation) => {
      const versionText = conversation.document_version_index
        ? `v${conversation.document_version_index}`
        : ""
      const docName = conversation.document_display_name || conversation.document_name || "多文档"
      const meta = conversation.document_version_index
        ? `${docName} ${versionText} · ${formatTime(conversation.updated_at)}`
        : `${docName} · ${formatTime(conversation.updated_at)}`
      const modeBadge = conversation.document_id
        ? '<span class="history-mode-badge">单文件</span>'
        : '<span class="history-mode-badge">全局</span>'
      const cleanTitle = formatConversationTitle(conversation.title) || docName
      const activeClass = conversation.id === state.activeConversationId ? "is-active" : ""
      return `
        <article class="history-item ${activeClass}" data-conversation-id="${conversation.id}">
          <span class="history-dot" aria-hidden="true"></span>
          <button class="history-item__main" type="button" data-conversation-load="${conversation.id}">
            <div class="history-item__title">
              <span class="history-item__name">${escapeHtml(cleanTitle)}</span>
              ${modeBadge}
            </div>
            <div class="history-item__meta">${escapeHtml(meta)}</div>
          </button>
          <div class="history-item__actions">
            <button class="history-action history-action--icon" type="button" data-conversation-delete="${conversation.id}" aria-label="删除">
              <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
                <path
                  d="M6.5 7.5v7m3.5-7v7m3.5-7v7M4 5.5h12m-8.5 0 .7-1.4A1 1 0 0 1 9.1 3.5h1.8a1 1 0 0 1 .9.6l.7 1.4m-7.2 0 .4 9.3a1.2 1.2 0 0 0 1.2 1.2h6.2a1.2 1.2 0 0 0 1.2-1.2l.4-9.3"
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="1.5"
                />
              </svg>
            </button>
          </div>
        </article>
      `
    })
    .join("")

  refs.conversationLibraryList.querySelectorAll("[data-conversation-load]").forEach((element) => {
    element.addEventListener("click", async () => {
      await loadConversation(Number(element.dataset.conversationLoad))
    })
  })

  refs.conversationLibraryList.querySelectorAll("[data-conversation-delete]").forEach((element) => {
    const triggerDelete = async (event) => {
      event.preventDefault()
      event.stopPropagation()
      if (state.isSubmitting) {
        return
      }
      await deleteConversation(Number(element.dataset.conversationDelete))
    }

    element.addEventListener("click", triggerDelete)
    element.addEventListener("keydown", async (event) => {
      if (event.key === "Enter" || event.key === " ") {
        await triggerDelete(event)
      }
    })
  })
}

function renderWorkspace() {
  const hasMessages = state.messages.length > 0
  syncWorkspaceChrome()

  if (!hasMessages) {
    shouldAutoScroll = true
    refs.messageList.innerHTML = ""
    return
  }

  refs.messageList.innerHTML = state.messages.map((message) => renderMessage(message)).join("")
}

function renderAttachmentTray() {
  if (!state.pendingImages.length) {
    refs.attachmentTray.className = "attachment-tray"
    refs.attachmentTray.innerHTML = ""
    return
  }

  refs.attachmentTray.className = "attachment-tray has-items"
  refs.attachmentTray.innerHTML = state.pendingImages
    .map(
      (image) => `
        <div class="attachment-chip">
          <span class="attachment-chip__name">${escapeHtml(image.name)}</span>
          <span class="attachment-chip__tag">待接入</span>
          <button class="attachment-chip__remove" type="button" data-image-id="${image.id}">移除</button>
        </div>
      `,
    )
    .join("")

  refs.attachmentTray.querySelectorAll("[data-image-id]").forEach((element) => {
    element.addEventListener("click", () => {
      state.pendingImages = state.pendingImages.filter((image) => image.id !== element.dataset.imageId)
      renderAttachmentTray()
    })
  })
}

function formatConversationTitle(title) {
  return String(title || "")
    .replace(/\s*\/\s*新对话\s*$/u, "")
    .trim()
}

function buildHeaderMeta(document, conversation, hasMessages) {
  if (!document) {
    return conversation ? "全局" : ""
  }

  const indexLabel = buildDocumentIndexLabel(document)

  if (!conversation) {
    return `${documentLabel(document)} · ${indexLabel}`
  }

  if (!isDocumentIndexReady(document)) {
    return `${documentLabel(document)} · ${indexLabel}`
  }

  if (!hasMessages) {
    return `${documentLabel(document)} · 0 条消息`
  }

  return `${documentLabel(document)} · ${state.messages.length} 条消息`
}

function buildBreadcrumb(document, conversation) {
  if (!document) {
    return formatConversationTitle(conversation?.title || "") || "问答"
  }

  const conversationLabel = formatConversationTitle(conversation?.title || "") || "问答"
  return `${document.file_name} › ${conversationLabel}`
}

function renderMessage(message) {
  if (message.role === "user") {
    return `
      <article class="message message--user ${message.pending ? "is-pending" : ""}" data-message-id="${message.id}">
        <div class="message-meta">你 · ${formatTime(message.created_at)}</div>
        <div class="user-bubble">${renderPlainText(message.content)}</div>
      </article>
    `
  }
  return `
    <article class="message message--assistant ${message.pending ? "is-pending" : ""}" data-message-id="${message.id}">
      <div class="message-meta">助手 · ${formatTime(message.created_at)}</div>
      <div class="assistant-flow">
        ${renderThinkingBlock(message)}
        ${renderAssistantBody(message)}
      </div>
    </article>
  `
}

function isThinkingExpanded(message) {
  if (typeof message?.thinking_expanded === "boolean") {
    return message.thinking_expanded
  }
  return Boolean(message?.pending)
}

function renderThinkingBlock(message) {
  const thinkingText = message.thinking_text || ""
  if (!message.pending && !thinkingText) {
    return ""
  }

  if (message.pending && !thinkingText && message.content) {
    return ""
  }

  const summaryText = message.pending ? "思考" : "思考"
  const expanded = isThinkingExpanded(message)
  const hintText = expanded ? "收起" : "展开"
  const body = message.pending && !thinkingText
    ? renderPendingWait(message)
    : renderPlainText(thinkingText)

  return `
    <details class="thinking-block" ${expanded ? "open" : ""}>
      <summary>
        <span class="thinking-tag">${summaryText}</span>
        <span class="thinking-hint">${hintText}</span>
      </summary>
      <div class="thinking-body">${body}</div>
    </details>
  `
}

function renderPendingWait(message) {
  return `
    <div class="thinking-wait">
      <span class="thinking-wait__line">${escapeHtml(buildPendingWaitBlockMessage(message))}</span>
      <span class="thinking-wait__dots" aria-hidden="true">
        <span></span><span></span><span></span>
      </span>
    </div>
  `
}

function buildPendingWaitBlockMessage(_message) {
  return "处理中"
}

function buildPendingWaitMessage(message) {
  const stage = message?.progress_stage || "正在准备回答"

  if (stage === "正在调用模型服务") {
    return "模型处理中"
  }

  if (stage === "正在整理文档上下文") {
    return "整理上下文"
  }

  if (stage === "正在读取文档页面缓存") {
    return "读取页面缓存"
  }

  if (stage === "正在检索 OCR 索引") {
    return "检索索引"
  }

  if (stage === "正在整理命中页上下文") {
    return "整理命中页"
  }

  if (stage === "模型已开始响应") {
    return "生成中"
  }

  return "处理中"
}

function syncPendingWaitTicker() {
  const hasAnimatedPending = state.messages.some(
    (message) => message.pending && !message.thinking_text && !message.content,
  )

  if (pendingWaitTickerId) {
    window.clearInterval(pendingWaitTickerId)
    pendingWaitTickerId = null
  }

  if (!hasAnimatedPending) {
    return
  }

  const pendingMessage = state.messages.find(
    (message) => message.pending && !message.thinking_text && !message.content,
  )
  if (pendingMessage) {
    setStatusLine(buildPendingWaitMessage(pendingMessage), "info")
  }
}

function renderAssistantBody(message) {
  if (message.pending) {
    if (message.content) {
      return `
        <div class="assistant-body assistant-stream">
          ${renderMarkdown(message.content, { renderMermaid: false })}<span class="stream-cursor">|</span>
        </div>
      `
    }

    return ""
  }

  return `
    <div class="assistant-body">${renderMarkdown(message.content || "", { renderMermaid: true })}</div>
  `
}

function patchUnclosedFences(content) {
  const fenceCount = (content.match(/```/g) || []).length
  if (fenceCount % 2 === 0) {
    return content
  }
  return `${content}\n\`\`\``
}

function scheduleMermaidRender() {
  if (!window.mermaid?.init || mermaidRenderQueued) {
    return
  }

  mermaidRenderQueued = true
  window.requestAnimationFrame(() => {
    mermaidRenderQueued = false

    document.querySelectorAll(".language-mermaid").forEach((el) => {
      if (el.dataset.mermaidRendered) {
        return
      }

      el.dataset.mermaidRendered = "1"
      const code = el.textContent || ""
      const pre = el.parentNode
      if (!pre?.parentNode) {
        return
      }

      const div = document.createElement("div")
      div.className = "mermaid"
      div.textContent = code
      pre.parentNode.replaceChild(div, pre)

      try {
        window.mermaid.init(undefined, div)
      } catch (error) {
        console.error(error)
      }
    })
  })
}

function renderMarkdown(content, options = {}) {
  const { renderMermaid = true } = options

  if (!content) {
    return "<p>暂无内容。</p>"
  }

  try {
    if (window.marked?.parse) {
      const patched = patchUnclosedFences(content)
      const html = window.marked.parse(patched)
      const safeHtml = window.DOMPurify?.sanitize ? window.DOMPurify.sanitize(html) : html
      const withCitations = injectPageCitations(safeHtml)

      if (renderMermaid && /```mermaid\b/i.test(patched)) {
        scheduleMermaidRender()
      }

      return withCitations
    }
  } catch (error) {
    console.error(error)
  }

  return injectPageCitations(renderPlainText(content))
}

function renderPlainText(content) {
  return escapeHtml(content || "").replace(/\n/g, "<br />")
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
}

function formatTime(value) {
  if (!value) {
    return "刚刚"
  }

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return "刚刚"
  }

  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date)
}

function autoResizeTextarea() {
  refs.questionInput.style.height = "auto"
  refs.questionInput.style.height = `${Math.min(refs.questionInput.scrollHeight, 220)}px`
}

function syncComposerState() {
  const activeDoc = currentDocument()
  const mode = currentMode()
  const readyToChat = hasAnyQueryableDocument()
  const hasQuestion = Boolean(refs.questionInput.value.trim())

  refs.questionInput.disabled = !readyToChat || state.isSubmitting
  refs.sendButton.disabled = !readyToChat || state.isSubmitting || !hasQuestion
  refs.globalChatButton.disabled = state.isSubmitting
  refs.singleChatButton.disabled = state.isSubmitting || !state.documents.length
  refs.uploadButton.disabled = state.isSubmitting
  refs.docsUploadButton.disabled = state.isSubmitting
  refs.imagePickerButton.disabled = state.isSubmitting

  if (readyToChat) {
    refs.questionInput.placeholder = "输入问题…"
    return
  }
  refs.questionInput.placeholder = activeDoc
    ? `${documentLabel(activeDoc)} ${buildDocumentIndexLabel(activeDoc)}`
    : "等待文档"
}

function scrollMessagesToBottom(force = false) {
  if (refs.workspace.dataset.mode !== "chat") {
    return
  }
  if (!force && !shouldAutoScroll) {
    return
  }
  refs.messageList.scrollTop = refs.messageList.scrollHeight
}

function isMessageListNearBottom() {
  const distanceToBottom =
    refs.messageList.scrollHeight - refs.messageList.scrollTop - refs.messageList.clientHeight
  return distanceToBottom <= AUTO_SCROLL_THRESHOLD
}
