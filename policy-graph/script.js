// =======================
// State Management
// =======================
let conversations = []
let currentConversationId = null
let selectedFiles = []
let currentTheme = 'dark'
let currentSimulationData = null
let chatHistory = []

// =======================
// Initialize App
// =======================
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide icons
    lucide.createIcons()
    
    // Load saved data
    loadConversations()
    loadTheme()
    
    // Setup event listeners
    setupEventListeners()
    
    // Initial render
    renderConversations()
})

// =======================
// Event Listeners Setup
// =======================
function setupEventListeners() {
    // Hamburger menu
    const hamburgerBtn = document.getElementById('hamburgerBtn')
    const sidebar = document.getElementById('sidebar')
    const backdrop = document.getElementById('sidebarBackdrop')
    
    hamburgerBtn.addEventListener('click', toggleSidebar)
    backdrop.addEventListener('click', closeSidebar)
    
    // New chat button
    document.getElementById('newChatBtn').addEventListener('click', () => {
        handleNewSimulation()
        closeSidebar()
    })
    
    // Theme toggle
    document.getElementById('themeToggle').addEventListener('click', toggleTheme)
    
    // Chat form (main simulation input)
    const chatForm = document.getElementById('chatForm')
    chatForm.addEventListener('submit', handleSubmit)
    
    // Message input
    const messageInput = document.getElementById('messageInput')
    messageInput.addEventListener('input', autoResizeTextarea)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            chatForm.dispatchEvent(new Event('submit'))
        }
    })
    
    // File upload
    const fileBtn = document.getElementById('fileBtn')
    const fileInput = document.getElementById('fileInput')
    
    fileBtn.addEventListener('click', () => fileInput.click())
    fileInput.addEventListener('change', handleFileSelect)
    
    // Drag and drop
    const inputWrapper = document.getElementById('inputWrapper')
    inputWrapper.addEventListener('dragover', handleDragOver)
    inputWrapper.addEventListener('dragleave', handleDragLeave)
    inputWrapper.addEventListener('drop', handleDrop)
    
    // Example prompts
    document.querySelectorAll('.example-card').forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.dataset.prompt
            document.getElementById('messageInput').value = prompt
            messageInput.dispatchEvent(new Event('input'))
        })
    })
    
    // New simulation button
    const newSimBtn = document.getElementById('newSimulationBtn')
    if (newSimBtn) {
        newSimBtn.addEventListener('click', handleNewSimulation)
    }
    
    // Chat assistant
    const chatAssistantForm = document.getElementById('chatAssistantForm')
    const chatAssistantInput = document.getElementById('chatAssistantInput')
    
    chatAssistantForm.addEventListener('submit', (e) => {
        e.preventDefault()
        sendChatMessage()
    })
    
    chatAssistantInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault()
            sendChatMessage()
        }
    })
    
    // Close chat sidebar
    const closeChatBtn = document.getElementById('closeChatBtn')
    closeChatBtn.addEventListener('click', toggleChatSidebar)
    
    // Floating chat button
    const floatingChatBtn = document.getElementById('floatingChatBtn')
    floatingChatBtn.addEventListener('click', openChatSidebar)
}

// =======================
// Sidebar Functions
// =======================
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar')
    const backdrop = document.getElementById('sidebarBackdrop')
    const hamburgerBtn = document.getElementById('hamburgerBtn')
    const icon = hamburgerBtn.querySelector('i')
    
    sidebar.classList.toggle('active')
    backdrop.classList.toggle('active')
    
    // Update icon
    if (sidebar.classList.contains('active')) {
        icon.setAttribute('data-lucide', 'x')
    } else {
        icon.setAttribute('data-lucide', 'menu')
    }
    lucide.createIcons()
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar')
    const backdrop = document.getElementById('sidebarBackdrop')
    const hamburgerBtn = document.getElementById('hamburgerBtn')
    const icon = hamburgerBtn.querySelector('i')
    
    sidebar.classList.remove('active')
    backdrop.classList.remove('active')
    
    icon.setAttribute('data-lucide', 'menu')
    lucide.createIcons()
}

// =======================
// Theme Functions
// =======================
function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark'
    currentTheme = savedTheme
    applyTheme(savedTheme)
}

function toggleTheme() {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark'
    applyTheme(currentTheme)
    localStorage.setItem('theme', currentTheme)
}

function applyTheme(theme) {
    const html = document.documentElement
    const icon = document.getElementById('themeIcon')
    const text = document.getElementById('themeText')
    
    if (theme === 'dark') {
        html.classList.add('dark')
        icon.setAttribute('data-lucide', 'sun')
        text.textContent = 'Light Mode'
    } else {
        html.classList.remove('dark')
        icon.setAttribute('data-lucide', 'moon')
        text.textContent = 'Dark Mode'
    }
    
    lucide.createIcons()
}

// =======================
// Conversation Functions
// =======================
function loadConversations() {
    const saved = localStorage.getItem('conversations')
    if (saved) {
        try {
            conversations = JSON.parse(saved)
        } catch (e) {
            console.error('Failed to load conversations:', e)
            conversations = []
        }
    }
}

function saveConversations() {
    localStorage.setItem('conversations', JSON.stringify(conversations))
}

function handleNewSimulation() {
    currentConversationId = null
    currentSimulationData = null
    chatHistory = []
    selectedFiles = []
    
    // Reset UI
    document.getElementById('messageInput').value = ''
    renderFileAttachments()
    showWelcomeScreen()
    closeChatSidebar()
    
    // Hide floating button on welcome screen
    document.getElementById('floatingChatBtn').classList.add('hidden')
    
    // Reset chat assistant
    const chatMessages = document.getElementById('chatMessagesContainer')
    chatMessages.innerHTML = `
        <div class="chat-message bot">
            <div class="chat-avatar bot">
                <i data-lucide="bot"></i>
            </div>
            <div class="chat-bubble bot">
                Hello! I'm ready to discuss the simulation results. Ask me anything about the analysis, graphs, or reports.
            </div>
        </div>
    `
    lucide.createIcons()
}

function selectConversation(id) {
    currentConversationId = id
    const conversation = conversations.find(c => c.id === id)
    
    if (conversation && conversation.simulationData) {
        currentSimulationData = conversation.simulationData
        chatHistory = conversation.chatHistory || []
        
        // Show results
        showResultsScreen()
        renderSimulationResults(conversation.simulationData)
        
        // Restore chat history
        restoreChatHistory()
        
        // Show floating button
        document.getElementById('floatingChatBtn').classList.remove('hidden')
    }
    
    renderConversations()
    closeSidebar()
}

function deleteConversation(id, event) {
    event.stopPropagation()
    
    conversations = conversations.filter(c => c.id !== id)
    
    if (currentConversationId === id) {
        handleNewSimulation()
    }
    
    saveConversations()
    renderConversations()
}

function renderConversations() {
    const container = document.getElementById('conversationsList')
    
    if (conversations.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i data-lucide="message-square" class="icon"></i>
                <p style="font-size: 0.875rem; color: var(--muted-foreground);">No simulations yet</p>
                <p style="font-size: 0.75rem; color: var(--muted-foreground); opacity: 0.6; margin-top: 0.25rem;">Start a new simulation to begin</p>
            </div>
        `
        lucide.createIcons()
        return
    }
    
    container.innerHTML = conversations.map((conv, index) => {
        const isActive = conv.id === currentConversationId
        const timeAgo = formatTimeAgo(new Date(conv.updatedAt))
        
        return `
            <div class="conversation-item ${isActive ? 'active' : ''}" 
                 onclick="selectConversation('${conv.id}')"
                 style="animation-delay: ${index * 0.03}s">
                <i data-lucide="activity"></i>
                <div class="conversation-info">
                    <div class="conversation-title">${escapeHtml(conv.title)}</div>
                    <div class="conversation-time">${timeAgo}</div>
                </div>
                <button class="delete-btn" onclick="deleteConversation('${conv.id}', event)">
                    <i data-lucide="trash-2"></i>
                </button>
            </div>
        `
    }).join('')
    
    lucide.createIcons()
}

// =======================
// Simulation Functions
// =======================
async function handleSubmit(e) {
    e.preventDefault()
    
    const input = document.getElementById('messageInput')
    const query = input.value.trim()
    
    if (!query && selectedFiles.length === 0) return
    
    // Show loading
    showLoading()
    
    // Prepare form data
    const formData = new FormData()
    if (query) formData.append('query', query)
    
    selectedFiles.forEach(file => {
        if (file.type === 'application/pdf') {
            formData.append('pdf', file)
        } else if (file.type.startsWith('image/')) {
            formData.append('image', file)
        }
    })
    
    try {
        // Make API call
        const response = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            body: formData
        })
        
        if (!response.ok) {
            const error = await response.json()
            throw new Error(error.detail || 'Server Error')
        }
        
        const data = await response.json()
        
        // Save simulation data
        currentSimulationData = data
        
        // Create or update conversation
        let conversation = conversations.find(c => c.id === currentConversationId)
        
        if (!conversation) {
            conversation = {
                id: Date.now().toString(),
                title: query.slice(0, 60) + (query.length > 60 ? '...' : ''),
                simulationData: data,
                chatHistory: [],
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString()
            }
            conversations.unshift(conversation)
            currentConversationId = conversation.id
        } else {
            conversation.simulationData = data
            conversation.updatedAt = new Date().toISOString()
        }
        
        saveConversations()
        renderConversations()
        
        // Clear input
        input.value = ''
        selectedFiles = []
        renderFileAttachments()
        autoResizeTextarea({ target: input })
        
        // Hide loading and show results
        hideLoading()
        showResultsScreen()
        renderSimulationResults(data)
        
        // Open chat sidebar and show floating button
        openChatSidebar()
        document.getElementById('floatingChatBtn').classList.remove('hidden')
        
    } catch (error) {
        hideLoading()
        alert('Simulation Error: ' + error.message)
        console.error('Simulation error:', error)
    }
}

function renderSimulationResults(data) {
    const container = document.getElementById('resultsContainer')
    
    if (!data || !data.reports) {
        container.innerHTML = '<p>No simulation data available.</p>'
        return
    }
    
    container.innerHTML = data.reports.map((report, index) => {
        const htmlContent = marked.parse(report.content)
        const graphId = `agent-cy-${index}`
        
        let graphHtml = ''
        if (report.diagram_json && report.diagram_json.length > 0) {
            graphHtml = `<div class="agent-graph-container" id="${graphId}"></div>`
        }
        
        return `
            <div class="agent-report-card" style="animation-delay: ${index * 0.1}s">
                <div class="agent-header">
                    <div class="agent-icon">
                        <i data-lucide="user-cog"></i>
                    </div>
                    <div class="agent-info">
                        <h3 class="agent-role">${escapeHtml(report.role)}</h3>
                        <p class="agent-description">Multi-agent analysis perspective</p>
                    </div>
                </div>
                <div class="agent-content">${htmlContent}</div>
                ${graphHtml}
            </div>
        `
    }).join('')
    
    lucide.createIcons()
    
    // Render graphs
    data.reports.forEach((report, index) => {
        if (report.diagram_json && report.diagram_json.length > 0) {
            setTimeout(() => {
                initCytoscape(`agent-cy-${index}`, report.diagram_json)
            }, 100)
        }
    })
}

function initCytoscape(containerId, elements) {
    const container = document.getElementById(containerId)
    if (!container) return
    
    cytoscape({
        container: container,
        elements: elements,
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': 'hsl(142.1, 70.6%, 45.3%)',
                    'label': 'data(label)',
                    'color': '#fff',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '10px',
                    'width': 'label',
                    'height': 30,
                    'padding': '10px',
                    'shape': 'round-rectangle',
                    'text-wrap': 'wrap',
                    'text-max-width': '80px',
                    'border-width': 2,
                    'border-color': '#fff'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': '#666',
                    'target-arrow-color': '#666',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'label': 'data(label)',
                    'font-size': '8px',
                    'color': '#999'
                }
            }
        ],
        layout: {
            name: 'cose',
            animate: false,
            padding: 30,
            componentSpacing: 50,
            nodeOverlap: 25,
            idealEdgeLength: 60
        }
    })
}

// =======================
// Chat Assistant Functions
// =======================
async function sendChatMessage() {
    const input = document.getElementById('chatAssistantInput')
    const message = input.value.trim()
    
    if (!message) return
    
    if (!currentSimulationData) {
        addChatBubble('Please run a simulation first so I have context!', 'bot')
        return
    }
    
    // Add user message to UI
    addChatBubble(message, 'user')
    input.value = ''
    
    // Add to history
    chatHistory.push({ role: 'user', content: message })
    
    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                history: chatHistory,
                context: JSON.stringify(currentSimulationData, null, 2)
            })
        })
        
        if (!response.ok) throw new Error('Chat Error')
        
        const data = await response.json()
        const reply = data.response
        
        // Add assistant message to UI
        addChatBubble(reply, 'bot')
        
        // Add to history
        chatHistory.push({ role: 'assistant', content: reply })
        
        // Update conversation
        const conversation = conversations.find(c => c.id === currentConversationId)
        if (conversation) {
            conversation.chatHistory = chatHistory
            conversation.updatedAt = new Date().toISOString()
            saveConversations()
        }
        
    } catch (error) {
        addChatBubble("Sorry, I'm having trouble connecting. Please try again.", 'bot')
        console.error('Chat error:', error)
    }
}

function addChatBubble(text, role) {
    const container = document.getElementById('chatMessagesContainer')
    const messageDiv = document.createElement('div')
    messageDiv.className = `chat-message ${role}`
    
    const avatarIcon = role === 'bot' ? 'bot' : 'user'
    
    messageDiv.innerHTML = `
        <div class="chat-avatar ${role}">
            <i data-lucide="${avatarIcon}"></i>
        </div>
        <div class="chat-bubble ${role}">
            ${escapeHtml(text)}
        </div>
    `
    
    container.appendChild(messageDiv)
    lucide.createIcons()
    
    // Scroll to bottom
    container.scrollTop = container.scrollHeight
}

function restoreChatHistory() {
    const container = document.getElementById('chatMessagesContainer')
    
    // Clear existing messages
    container.innerHTML = `
        <div class="chat-message bot">
            <div class="chat-avatar bot">
                <i data-lucide="bot"></i>
            </div>
            <div class="chat-bubble bot">
                Hello! I'm ready to discuss the simulation results. Ask me anything about the analysis, graphs, or reports.
            </div>
        </div>
    `
    
    // Add chat history
    chatHistory.forEach(msg => {
        addChatBubble(msg.content, msg.role === 'user' ? 'user' : 'bot')
    })
    
    lucide.createIcons()
}

function toggleChatSidebar() {
    const chatSidebar = document.getElementById('chatSidebar')
    const mainContent = document.querySelector('.main-content')
    const floatingBtn = document.getElementById('floatingChatBtn')
    
    const isOpen = chatSidebar.classList.contains('active')
    
    if (isOpen) {
        chatSidebar.classList.remove('active')
        mainContent.classList.remove('chat-open')
        floatingBtn.classList.remove('hidden')
    } else {
        chatSidebar.classList.add('active')
        mainContent.classList.add('chat-open')
        floatingBtn.classList.add('hidden')
    }
}

function openChatSidebar() {
    const chatSidebar = document.getElementById('chatSidebar')
    const mainContent = document.querySelector('.main-content')
    const floatingBtn = document.getElementById('floatingChatBtn')
    
    chatSidebar.classList.add('active')
    mainContent.classList.add('chat-open')
    floatingBtn.classList.add('hidden')
}

function closeChatSidebar() {
    const chatSidebar = document.getElementById('chatSidebar')
    const mainContent = document.querySelector('.main-content')
    const floatingBtn = document.getElementById('floatingChatBtn')
    
    chatSidebar.classList.remove('active')
    mainContent.classList.remove('chat-open')
    floatingBtn.classList.remove('hidden')
}

// =======================
// File Handling
// =======================
function handleFileSelect(e) {
    const files = Array.from(e.target.files)
    const validFiles = files.filter(f => 
        f.type === 'application/pdf' || f.type === 'text/plain'
    )
    
    selectedFiles = [...selectedFiles, ...validFiles]
    renderFileAttachments()
    e.target.value = '' // Reset input
}

function handleDragOver(e) {
    e.preventDefault()
    e.currentTarget.classList.add('dragging')
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragging')
}

function handleDrop(e) {
    e.preventDefault()
    e.currentTarget.classList.remove('dragging')
    
    const files = Array.from(e.dataTransfer.files)
    const validFiles = files.filter(f => 
        f.type === 'application/pdf' || f.type === 'text/plain'
    )
    
    selectedFiles = [...selectedFiles, ...validFiles]
    renderFileAttachments()
}

function removeFile(index) {
    selectedFiles.splice(index, 1)
    renderFileAttachments()
}

function renderFileAttachments() {
    const container = document.getElementById('fileAttachments')
    
    if (selectedFiles.length === 0) {
        container.classList.add('hidden')
        return
    }
    
    container.classList.remove('hidden')
    container.innerHTML = selectedFiles.map((file, index) => `
        <div class="file-attachment">
            <i data-lucide="file-text"></i>
            <span>${escapeHtml(file.name)}</span>
            <button class="file-remove" onclick="removeFile(${index})">
                <i data-lucide="x"></i>
            </button>
        </div>
    `).join('')
    
    lucide.createIcons()
}

// =======================
// UI Helper Functions
// =======================
function showWelcomeScreen() {
    const welcomeScreen = document.getElementById('welcomeScreen')
    const resultsScreen = document.getElementById('resultsScreen')
    
    welcomeScreen.classList.remove('hidden')
    resultsScreen.classList.add('hidden')
}

function showResultsScreen() {
    const welcomeScreen = document.getElementById('welcomeScreen')
    const resultsScreen = document.getElementById('resultsScreen')
    
    welcomeScreen.classList.add('hidden')
    resultsScreen.classList.remove('hidden')
}

function showLoading() {
    const overlay = document.getElementById('loadingOverlay')
    overlay.classList.remove('hidden')
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay')
    overlay.classList.add('hidden')
}

function autoResizeTextarea(e) {
    const textarea = e.target
    textarea.style.height = 'auto'
    textarea.style.height = Math.min(textarea.scrollHeight, 128) + 'px'
}

// =======================
// Utility Functions
// =======================
function formatTimeAgo(date) {
    const now = new Date()
    const diffMs = now - date
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)
    
    if (diffMins < 1) return 'just now'
    if (diffMins < 60) return `${diffMins} min${diffMins > 1 ? 's' : ''} ago`
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} week${Math.floor(diffDays / 7) > 1 ? 's' : ''} ago`
    return `${Math.floor(diffDays / 30)} month${Math.floor(diffDays / 30) > 1 ? 's' : ''} ago`
}

function escapeHtml(text) {
    const div = document.createElement('div')
    div.textContent = text
    return div.innerHTML
}