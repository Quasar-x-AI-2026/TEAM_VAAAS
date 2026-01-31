// --- STATE ---
let selectedFile = null;
let currentSimulationData = null; // Store for chatbot
let chatHistory = [];

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    // Check for "Enter" key in chat input
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendChatMessage();
        });
    }
});

// --- UI INTERACTIONS ---

function handleFileSelect(input) {
    if (input.files.length > 0) {
        selectedFile = input.files[0];
        const badge = document.getElementById('fileBadge');
        badge.innerText = selectedFile.name;
        badge.classList.remove('hidden');
    }
}

function resetView() {
    document.getElementById('resultsSection').style.opacity = '0';
    setTimeout(() => {
        document.getElementById('resultsSection').style.display = 'none';

        const hero = document.getElementById('heroSection');
        hero.style.display = 'flex';
        setTimeout(() => hero.style.opacity = '1', 10);

        // Clear inputs
        document.getElementById('queryInput').value = '';
        selectedFile = null;
        document.getElementById('fileInput').value = '';
        document.getElementById('fileBadge').classList.add('hidden');

        // Reset Chat
        chatHistory = [];
        currentSimulationData = null;
        const chatMsgs = document.getElementById('chatMessages');
        if (chatMsgs) {
            chatMsgs.innerHTML = '<div class="chat-message bot">Hello! I am ready to discuss the simulation results. Ask me anything about the graphs or reports.</div>';
        }

    }, 500);
}

function showResults() {
    const hero = document.getElementById('heroSection');
    hero.style.opacity = '0';
    setTimeout(() => {
        hero.style.display = 'none';

        const results = document.getElementById('resultsSection');
        results.style.display = 'block';
        setTimeout(() => results.style.opacity = '1', 10);
    }, 500);
}

// --- API HANDLING ---

document.getElementById('simForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const query = document.getElementById('queryInput').value;
    if (!query && !selectedFile) return;

    // Show Loader
    document.getElementById('loader').style.display = 'flex';

    // Prepare Data
    const formData = new FormData();
    if (query) formData.append('query', query);

    if (selectedFile) {
        if (selectedFile.type.includes('pdf')) {
            formData.append('pdf', selectedFile);
        } else if (selectedFile.type.includes('image')) {
            formData.append('image', selectedFile);
        }
    }

    try {
        const response = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server Error");
        }

        const data = await response.json();
        currentSimulationData = JSON.stringify(data, null, 2); // Save for Chat

        // Hide loader and switch view
        document.getElementById('loader').style.display = 'none';
        showResults();

        renderAgentsWithGraphs(data.reports);

    } catch (error) {
        document.getElementById('loader').style.display = 'none';
        alert("Simulation Error: " + error.message);
    }
});

function renderAgentsWithGraphs(reports) {
    const agentsGrid = document.getElementById('agentsGrid');
    agentsGrid.innerHTML = '';

    if (!reports) return;

    reports.forEach((report, index) => {
        const htmlContent = marked.parse(report.content);
        const card = document.createElement('div');
        card.className = 'agent-report';

        // Unique ID for this agent's graph
        const graphId = `agent-cy-${index}`;

        let graphHtml = '';
        if (report.diagram_json && report.diagram_json.length > 0) {
            graphHtml = `<div class="agent-graph-container" id="${graphId}"></div>`;
        }

        card.innerHTML = `
            <h4><i class="fa-solid fa-user-astronaut"></i> ${report.role}</h4>
            <div class="content">${htmlContent}</div>
            ${graphHtml}
        `;
        agentsGrid.appendChild(card);

        // Render Graph if data exists
        if (report.diagram_json && report.diagram_json.length > 0) {
            console.log(`[Graph] Rendering for ${report.role}:`, report.diagram_json);
            // Need a slight timeout to ensure DOM is ready
            setTimeout(() => {
                initCytoscape(graphId, report.diagram_json);
            }, 100);
        } else {
            console.warn(`[Graph] No diagram data for ${report.role}`);
        }
    });
}

function initCytoscape(containerId, elements) {
    cytoscape({
        container: document.getElementById(containerId),
        elements: elements,
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': '#10b981', // Agent Green
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
                    'border-width': 1,
                    'border-color': '#fff'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': '#555',
                    'target-arrow-color': '#555',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            }
        ],
        layout: {
            name: 'cose',
            animate: false,
            padding: 20,
            componentSpacing: 40,
            nodeOverlap: 20,
            idealEdgeLength: 50
        }
    });
}

// --- CHATBOT LOGIC ---

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if (!msg) return;

    if (!currentSimulationData) {
        addBubble("Please run a simulation first so I have context!", 'bot');
        return;
    }

    // UI Updates
    addBubble(msg, 'user');
    input.value = '';

    // Add User msg to history
    chatHistory.push({ role: "user", content: msg });

    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                history: chatHistory,
                context: currentSimulationData
            })
        });

        if (!response.ok) throw new Error("Chat Error");

        const data = await response.json();
        const reply = data.response;

        addBubble(reply, 'bot');
        chatHistory.push({ role: "assistant", content: reply });

    } catch (e) {
        addBubble("Sorry, I'm having trouble connecting to the brain.", 'bot');
    }
}

function addBubble(text, role) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `chat-message ${role}`;
    div.innerText = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}
