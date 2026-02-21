// ── Particle Background ──────────────────────────────
(function initParticles() {
  const canvas = document.getElementById("particles");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let w, h, particles;
  const COUNT = 50;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }

  function createParticles() {
    particles = [];
    for (let i = 0; i < COUNT; i++) {
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.3 + 0.05,
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(51, 153, 255, ${p.alpha})`;
      ctx.fill();
    }

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(51, 153, 255, ${0.06 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  resize();
  createParticles();
  draw();
  window.addEventListener("resize", () => { resize(); createParticles(); });
})();

// ── DOM Elements ──────────────────────────────────────
const chatMessages = document.getElementById("chatMessages");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");

let isLoading = false;

// ── Time Formatter ────────────────────────────────────
function timeNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ── Markdown-like Formatting ──────────────────────────
function formatText(text) {
  return text
    // Headers
    .replace(/^### (.+)$/gm, '<h4 style="margin:8px 0 4px;font-size:13px;color:#94A3B8">$1</h4>')
    .replace(/^## (.+)$/gm, '<h3 style="margin:10px 0 4px;font-size:14px">$1</h3>')
    // Bold
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    // Italic
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    // Inline code
    .replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:4px;font-size:12px">$1</code>')
    // Bullet points
    .replace(/^- (.+)$/gm, '<div style="padding-left:12px;margin:2px 0">• $1</div>')
    // Numbered lists
    .replace(/^(\d+)\. (.+)$/gm, '<div style="padding-left:12px;margin:2px 0">$1. $2</div>')
    // Source citations
    .replace(/\(Source: ([^)]+)\)/g, '<span style="color:#3399FF;font-size:12px;opacity:0.85">(📄 $1)</span>')
    // Line breaks
    .replace(/\n/g, "<br>");
}

// ── Add Message ───────────────────────────────────────
function addMessage(text, sender, extra = "") {
  // Remove welcome screen on first message
  const welcome = chatMessages.querySelector(".welcome");
  if (welcome) welcome.remove();

  const msg = document.createElement("div");
  msg.className = `message ${sender}`;

  const avatarText = sender === "user" ? "You" : "AI";
  const timeStr = extra ? `${timeNow()} · ${extra}` : timeNow();

  msg.innerHTML = `
    <div class="msg-avatar">${avatarText === "You" ? "👤" : "🤖"}</div>
    <div>
      <div class="msg-bubble">${formatText(text)}</div>
      <span class="msg-time">${timeStr}</span>
    </div>
  `;

  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── Typing Indicator ──────────────────────────────────
function showTyping() {
  const typing = document.createElement("div");
  typing.className = "message bot";
  typing.id = "typing";
  typing.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div>
      <div class="msg-bubble">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>
  `;
  chatMessages.appendChild(typing);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById("typing");
  if (el) el.remove();
}

// ── Send Question ─────────────────────────────────────
async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question || isLoading) return;

  isLoading = true;
  sendBtn.disabled = true;
  questionInput.value = "";

  addMessage(question, "user");
  showTyping();

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();
    hideTyping();

    if (data.error) {
      addMessage("Sorry, something went wrong. Please try again.", "bot");
    } else {
      addMessage(data.answer, "bot", `${data.time}s`);
    }
  } catch (err) {
    hideTyping();
    addMessage("Connection error. Please check if the server is running.", "bot");
  }

  isLoading = false;
  sendBtn.disabled = false;
  questionInput.focus();
}

// ── Event Listeners ───────────────────────────────────
sendBtn.addEventListener("click", sendQuestion);

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

// ── Suggestion Chips ──────────────────────────────────
document.addEventListener("click", (e) => {
  if (e.target.classList.contains("suggestion-chip")) {
    questionInput.value = e.target.textContent.trim();
    sendQuestion();
  }
});
