/**
 * VerifiQ - High Performance Canvas Particle Physics & Network Simulation
 * Pure Vanilla ES6+ Canvas 2D Engine with Zero Dependencies.
 */

// Canvas & Context Setup
const canvas = document.getElementById('simulation-canvas');
const ctx = canvas.getContext('2d');

// UI DOM Elements
const dashboard = document.getElementById('dashboard-main');
const btnToggleUI = document.getElementById('btn-toggle-ui');
const toggleUIText = document.getElementById('toggle-ui-text');
const btnSound = document.getElementById('btn-sound');
const soundIcon = document.getElementById('sound-icon');
const soundText = document.getElementById('sound-text');
const fpsValEl = document.getElementById('fps-val');

const countRealEl = document.getElementById('count-real');
const countFakeEl = document.getElementById('count-fake');
const countUnverifiedEl = document.getElementById('count-unverified');

const barRealEl = document.getElementById('bar-real');
const barFakeEl = document.getElementById('bar-fake');
const barUnverifiedEl = document.getElementById('bar-unverified');

const interactionSelect = document.getElementById('control-interaction');
const toggleMesh = document.getElementById('toggle-mesh');
const densitySlider = document.getElementById('control-density');
const densityVal = document.getElementById('density-val');
const speedSlider = document.getElementById('control-speed');
const speedVal = document.getElementById('speed-val');

const presetBtns = document.querySelectorAll('.preset-btn');
const resetBtn = document.getElementById('btn-reset');
const injectBtn = document.getElementById('btn-inject');
const pulseBtn = document.getElementById('btn-pulse');

// Simulation State
let particles = [];
let shockwaves = [];
let maxParticles = parseInt(densitySlider.value, 10) || 150;
let globalSpeedMultiplier = parseFloat(speedSlider.value) || 1.0;
let meshEnabled = toggleMesh ? toggleMesh.checked : true;
let soundEnabled = false;
let audioCtx = null;

let dpr = window.devicePixelRatio || 1;
let width = window.innerWidth;
let height = window.innerHeight;

// Pointer State
const pointer = {
    x: -1000,
    y: -1000,
    active: false,
    radius: 130
};

// Theme Colors
const COLORS = {
    real: '#10B981',             // Emerald
    realTail: 'rgba(16, 185, 129, 0.12)',
    fake: '#F43F5E',             // Rose / Crimson
    fakeTail: 'rgba(244, 63, 94, 0.12)',
    unverified: '#F59E0B',       // Amber
    unverifiedTail: 'rgba(245, 158, 11, 0.12)',
    meshLine: 'rgba(255, 255, 255, 0.05)',
    cyan: '#38BDF8'
};

// Web Audio API Synthesizer (Native, Zero Dependency)
function initAudio() {
    if (!audioCtx) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) {
            audioCtx = new AudioContextClass();
        }
    }
    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
}

function playTone(freq, type = 'sine', duration = 0.12, gainLevel = 0.08) {
    if (!soundEnabled || !audioCtx) return;
    try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(gainLevel, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch {
        // Suppress audio failure if context state blocks
    }
}

function playBurstSound() {
    if (!soundEnabled || !audioCtx) return;
    try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(220, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(70, audioCtx.currentTime + 0.25);
        gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.25);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.25);
    } catch {}
}

function playPulseSound() {
    if (!soundEnabled || !audioCtx) return;
    try {
        [523.25, 659.25, 783.99].forEach((freq, idx) => {
            setTimeout(() => playTone(freq, 'triangle', 0.25, 0.07), idx * 60);
        });
    } catch {}
}

// Particle Class
class Particle {
    constructor(x, y, type = null) {
        this.x = x !== undefined ? x : Math.random() * width;
        this.y = y !== undefined ? y : Math.random() * height;
        
        const speed = Math.random() * 1.4 + 0.6;
        const angle = Math.random() * Math.PI * 2;
        this.vx = Math.cos(angle) * speed;
        this.vy = Math.sin(angle) * speed;
        
        if (type) {
            this.type = type;
        } else {
            const rand = Math.random();
            if (rand < 0.45) this.type = 'real';
            else if (rand < 0.75) this.type = 'fake';
            else this.type = 'unverified';
        }
        
        this.radius = Math.random() * 2.5 + 3.0;
        if (this.type === 'fake') {
            this.radius += 1.2;
        }
        this.mass = this.radius;
    }

    update() {
        this.x += this.vx * globalSpeedMultiplier;
        this.y += this.vy * globalSpeedMultiplier;

        // Fluid friction damping
        this.vx *= 0.992;
        this.vy *= 0.992;

        // Boundary bounce
        if (this.x - this.radius < 0) {
            this.x = this.radius;
            this.vx = Math.abs(this.vx) * 0.9;
        } else if (this.x + this.radius > width) {
            this.x = width - this.radius;
            this.vx = -Math.abs(this.vx) * 0.9;
        }

        if (this.y - this.radius < 0) {
            this.y = this.radius;
            this.vy = Math.abs(this.vy) * 0.9;
        } else if (this.y + this.radius > height) {
            this.y = height - this.radius;
            this.vy = -Math.abs(this.vy) * 0.9;
        }

        // Pointer vector physics
        if (pointer.active) {
            const dx = this.x - pointer.x;
            const dy = this.y - pointer.y;
            const dist = Math.hypot(dx, dy);

            if (dist < pointer.radius && dist > 0.1) {
                const force = (pointer.radius - dist) / pointer.radius;
                const angle = Math.atan2(dy, dx);
                const mode = interactionSelect.value;
                
                if (mode === 'repel') {
                    this.vx += Math.cos(angle) * force * 1.2;
                    this.vy += Math.sin(angle) * force * 1.2;
                } else if (mode === 'attract') {
                    this.vx -= Math.cos(angle) * force * 1.1;
                    this.vy -= Math.sin(angle) * force * 1.1;
                } else if (mode === 'quarantine') {
                    // Pull and purify
                    this.vx -= Math.cos(angle) * force * 0.6;
                    this.vy -= Math.sin(angle) * force * 0.6;
                    if (this.type !== 'real' && Math.random() < 0.05) {
                        this.type = 'real';
                    }
                }
            }
        }
    }

    draw() {
        // Outer halo
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius + 3.5, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[this.type + 'Tail'];
        ctx.fill();

        // Core particle
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[this.type];
        ctx.fill();
    }
}

// Canvas Resizing
function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    width = window.innerWidth;
    height = window.innerHeight;

    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.scale(dpr, dpr);
}

// Particle Initialization
function initParticles() {
    particles = [];
    shockwaves = [];
    for (let i = 0; i < maxParticles; i++) {
        particles.push(new Particle());
    }
}

// Event Listeners
function setupEventListeners() {
    window.addEventListener('resize', () => {
        resizeCanvas();
        particles.forEach(p => {
            if (p.x > width) p.x = Math.random() * width;
            if (p.y > height) p.y = Math.random() * height;
        });
    });

    const updatePointerPos = (e) => {
        pointer.x = e.clientX;
        pointer.y = e.clientY;
    };

    canvas.addEventListener('pointerdown', (e) => {
        pointer.active = true;
        updatePointerPos(e);
        initAudio();

        // Trigger Shockwave ripple
        triggerShockwave(pointer.x, pointer.y, interactionSelect.value === 'quarantine' ? '#10B981' : '#38BDF8');

        if (interactionSelect.value === 'spawn') {
            for (let i = 0; i < 6; i++) {
                particles.push(new Particle(pointer.x + (Math.random() - 0.5) * 20, pointer.y + (Math.random() - 0.5) * 20));
            }
            if (particles.length > maxParticles * 1.5) {
                particles.splice(0, particles.length - Math.round(maxParticles * 1.5));
            }
            playTone(440, 'sine', 0.08, 0.05);
        }
    });

    canvas.addEventListener('pointermove', updatePointerPos);
    canvas.addEventListener('pointerup', () => { pointer.active = false; });
    canvas.addEventListener('pointercancel', () => { pointer.active = false; });
    canvas.addEventListener('pointerleave', () => { 
        pointer.active = false; 
        pointer.x = -1000;
        pointer.y = -1000;
    });

    // Prevent gestures on canvas
    canvas.addEventListener('touchstart', (e) => e.preventDefault(), { passive: false });
    canvas.addEventListener('touchmove', (e) => e.preventDefault(), { passive: false });

    // UI Drawer Toggle
    if (btnToggleUI && dashboard) {
        btnToggleUI.addEventListener('click', () => {
            const isCollapsed = dashboard.classList.toggle('collapsed');
            btnToggleUI.setAttribute('aria-expanded', !isCollapsed);
            if (isCollapsed) {
                toggleUIText.textContent = 'Show Panel';
                btnToggleUI.classList.remove('active');
            } else {
                toggleUIText.textContent = 'Hide Panel';
                btnToggleUI.classList.add('active');
            }
            playTone(600, 'sine', 0.05, 0.03);
        });
    }

    // Sound Toggle
    if (btnSound) {
        btnSound.addEventListener('click', () => {
            initAudio();
            soundEnabled = !soundEnabled;
            btnSound.setAttribute('aria-pressed', soundEnabled);
            if (soundEnabled) {
                soundIcon.textContent = '🔊';
                soundText.textContent = 'Sound: On';
                btnSound.classList.add('active');
                playTone(523.25, 'triangle', 0.15, 0.08);
            } else {
                soundIcon.textContent = '🔇';
                soundText.textContent = 'Sound: Off';
                btnSound.classList.remove('active');
            }
        });
    }

    // Keyboard Shortcuts (WCAG accessibility & power users)
    window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        if (e.key === 'h' || e.key === 'H') {
            e.preventDefault();
            btnToggleUI.click();
        } else if (e.key === ' ' || e.code === 'Space') {
            e.preventDefault();
            injectBtn.click();
        } else if (e.key === 'f' || e.key === 'F') {
            e.preventDefault();
            pulseBtn.click();
        } else if (e.key === 'm' || e.key === 'M') {
            e.preventDefault();
            toggleMesh.checked = !toggleMesh.checked;
            meshEnabled = toggleMesh.checked;
            playTone(700, 'sine', 0.06, 0.03);
        } else if (e.key === 'r' || e.key === 'R') {
            e.preventDefault();
            resetBtn.click();
        }
    });
}

// Shockwave Trigger
function triggerShockwave(x, y, color = '#38BDF8') {
    shockwaves.push({
        x,
        y,
        radius: 2,
        maxRadius: 180,
        speed: 5.5,
        alpha: 0.9,
        color
    });
}

// Update Shockwaves
function updateShockwaves() {
    for (let i = shockwaves.length - 1; i >= 0; i--) {
        const sw = shockwaves[i];
        sw.radius += sw.speed;
        sw.alpha = Math.max(0, 1.0 - sw.radius / sw.maxRadius);

        // Apply physical impulse to particles in shockwave wavefront
        particles.forEach(p => {
            const dx = p.x - sw.x;
            const dy = p.y - sw.y;
            const dist = Math.hypot(dx, dy);
            if (Math.abs(dist - sw.radius) < 18 && dist > 0.1) {
                const impulse = 0.7 * sw.alpha;
                p.vx += (dx / dist) * impulse;
                p.vy += (dy / dist) * impulse;
            }
        });

        // Draw expanding ring
        ctx.beginPath();
        ctx.arc(sw.x, sw.y, sw.radius, 0, Math.PI * 2);
        ctx.strokeStyle = sw.color === '#10B981' 
            ? `rgba(16, 185, 129, ${sw.alpha * 0.5})` 
            : `rgba(56, 189, 248, ${sw.alpha * 0.5})`;
        ctx.lineWidth = 2;
        ctx.stroke();

        if (sw.radius >= sw.maxRadius) {
            shockwaves.splice(i, 1);
        }
    }
}

// Update UI Statistics
function updateStats() {
    let real = 0, fake = 0, unverified = 0;
    particles.forEach(p => {
        if (p.type === 'real') real++;
        else if (p.type === 'fake') fake++;
        else unverified++;
    });

    countRealEl.textContent = real;
    countFakeEl.textContent = fake;
    countUnverifiedEl.textContent = unverified;

    const total = particles.length || 1;
    const pReal = ((real / total) * 100).toFixed(1);
    const pFake = ((fake / total) * 100).toFixed(1);
    const pUnv = (100 - parseFloat(pReal) - parseFloat(pFake)).toFixed(1);

    if (barRealEl && barFakeEl && barUnverifiedEl) {
        barRealEl.style.width = `${pReal}%`;
        barFakeEl.style.width = `${pFake}%`;
        barUnverifiedEl.style.width = `${pUnv}%`;
    }
}

// Draw Network Mesh Filaments
function drawNetworkMesh() {
    if (!meshEnabled) return;
    const maxMeshDist = 65;
    const maxMeshDistSq = maxMeshDist * maxMeshDist;

    for (let i = 0; i < particles.length; i++) {
        const p1 = particles[i];
        for (let j = i + 1; j < particles.length; j++) {
            const p2 = particles[j];
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const distSq = dx * dx + dy * dy;

            if (distSq < maxMeshDistSq) {
                const dist = Math.sqrt(distSq);
                const alpha = (1 - dist / maxMeshDist) * 0.22;
                
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                
                if (p1.type === 'real' && p2.type === 'real') {
                    ctx.strokeStyle = `rgba(16, 185, 129, ${alpha * 1.5})`;
                } else if (p1.type === 'fake' && p2.type === 'fake') {
                    ctx.strokeStyle = `rgba(244, 63, 94, ${alpha * 1.5})`;
                } else {
                    ctx.strokeStyle = `rgba(148, 163, 184, ${alpha * 0.8})`;
                }
                
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }
    }
}

// Particle Collisions and Contagion Dynamics
function handleCollisions() {
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const p1 = particles[i];
            const p2 = particles[j];

            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const minDist = p1.radius + p2.radius;

            if (Math.abs(dx) > minDist || Math.abs(dy) > minDist) continue;

            const distSq = dx * dx + dy * dy;
            if (distSq >= minDist * minDist || distSq === 0) continue;

            const dist = Math.sqrt(distSq);
            const angle = Math.atan2(dy, dx);
            const sin = Math.sin(angle);
            const cos = Math.cos(angle);

            // Rotate velocities
            const vx1 = p1.vx * cos + p1.vy * sin;
            const vy1 = p1.vy * cos - p1.vx * sin;
            const vx2 = p2.vx * cos + p2.vy * sin;
            const vy2 = p2.vy * cos - p2.vx * sin;

            // Elastic collision resolution
            const vx1Final = ((p1.mass - p2.mass) * vx1 + 2 * p2.mass * vx2) / (p1.mass + p2.mass);
            const vx2Final = ((p2.mass - p1.mass) * vx2 + 2 * p1.mass * vx1) / (p1.mass + p2.mass);

            p1.vx = vx1Final * cos - vy1 * sin;
            p1.vy = vy1 * cos + vx1Final * sin;
            p2.vx = vx2Final * cos - vy2 * sin;
            p2.vy = vy2 * cos + vx2Final * sin;

            // Separation to prevent sticking
            const overlap = minDist - dist;
            const sepX = Math.cos(angle) * (overlap / 2);
            const sepY = Math.sin(angle) * (overlap / 2);
            p1.x -= sepX;
            p1.y -= sepY;
            p2.x += sepX;
            p2.y += sepY;

            // Contagion & Verification Dynamics
            if (p1.type === 'fake' && p2.type === 'unverified' && Math.random() < 0.20) {
                p2.type = 'fake';
            } else if (p2.type === 'fake' && p1.type === 'unverified' && Math.random() < 0.20) {
                p1.type = 'fake';
            }

            if (p1.type === 'real' && p2.type === 'unverified' && Math.random() < 0.28) {
                p2.type = 'real';
            } else if (p2.type === 'real' && p1.type === 'unverified' && Math.random() < 0.28) {
                p1.type = 'real';
            }

            // Real vs Fake contention
            if ((p1.type === 'real' && p2.type === 'fake') || (p1.type === 'fake' && p2.type === 'real')) {
                if (Math.random() < 0.08) {
                    if (p1.type === 'fake') p1.type = 'unverified';
                    if (p2.type === 'fake') p2.type = 'unverified';
                }
            }
        }
    }
}

// Animation Loop & FPS Telemetry
let lastFrameTime = performance.now();
let frameCount = 0;

function animate(currentTime = performance.now()) {
    ctx.clearRect(0, 0, width, height);

    // Measure FPS smoothly
    frameCount++;
    if (currentTime - lastFrameTime >= 500) {
        const currentFps = Math.round((frameCount * 1000) / (currentTime - lastFrameTime));
        if (fpsValEl) fpsValEl.textContent = currentFps;
        frameCount = 0;
        lastFrameTime = currentTime;
    }

    // Physics & Collisions
    handleCollisions();
    drawNetworkMesh();
    updateShockwaves();

    particles.forEach(p => {
        p.update();
        p.draw();
    });

    // Draw pointer vector field
    if (pointer.active && pointer.x > 0 && interactionSelect.value !== 'spawn') {
        ctx.beginPath();
        ctx.arc(pointer.x, pointer.y, pointer.radius, 0, Math.PI * 2);
        ctx.strokeStyle = interactionSelect.value === 'quarantine' 
            ? 'rgba(16, 185, 129, 0.2)' 
            : 'rgba(56, 189, 248, 0.2)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath();
        ctx.arc(pointer.x, pointer.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = interactionSelect.value === 'quarantine' ? '#10B981' : '#38BDF8';
        ctx.fill();
    }

    updateStats();
    requestAnimationFrame(animate);
}

// UI Controls Hookup
if (densitySlider) {
    densitySlider.addEventListener('input', (e) => {
        maxParticles = parseInt(e.target.value, 10);
        if (densityVal) densityVal.textContent = maxParticles;
        densitySlider.setAttribute('aria-valuenow', maxParticles);

        if (particles.length < maxParticles) {
            while (particles.length < maxParticles) {
                particles.push(new Particle());
            }
        } else {
            particles.splice(maxParticles);
        }
    });
}

if (speedSlider) {
    speedSlider.addEventListener('input', (e) => {
        globalSpeedMultiplier = parseFloat(e.target.value);
        if (speedVal) speedVal.textContent = `${globalSpeedMultiplier.toFixed(1)}x`;
        speedSlider.setAttribute('aria-valuenow', globalSpeedMultiplier.toFixed(1));
    });
}

if (toggleMesh) {
    toggleMesh.addEventListener('change', (e) => {
        meshEnabled = e.target.checked;
    });
}

// Preset Buttons
presetBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
        initAudio();
        const preset = e.target.getAttribute('data-preset');
        if (preset === 'equilibrium') {
            particles.forEach((p, idx) => {
                p.type = idx % 3 === 0 ? 'real' : (idx % 3 === 1 ? 'fake' : 'unverified');
            });
            globalSpeedMultiplier = 1.0;
            speedSlider.value = 1.0;
            speedVal.textContent = '1.0x';
            triggerShockwave(width / 2, height / 2, '#38BDF8');
            playTone(440, 'sine', 0.15, 0.05);
        } else if (preset === 'outbreak') {
            particles.forEach(p => {
                p.type = Math.random() < 0.75 ? 'fake' : 'unverified';
            });
            globalSpeedMultiplier = 1.4;
            speedSlider.value = 1.4;
            speedVal.textContent = '1.4x';
            triggerShockwave(width / 2, height / 2, '#F43F5E');
            playBurstSound();
        } else if (preset === 'verified') {
            particles.forEach(p => {
                p.type = Math.random() < 0.80 ? 'real' : 'unverified';
            });
            globalSpeedMultiplier = 0.9;
            speedSlider.value = 0.9;
            speedVal.textContent = '0.9x';
            triggerShockwave(width / 2, height / 2, '#10B981');
            playPulseSound();
        }
    });
});

if (resetBtn) {
    resetBtn.addEventListener('click', () => {
        initAudio();
        initParticles();
        playTone(330, 'triangle', 0.1, 0.04);
    });
}

if (injectBtn) {
    injectBtn.addEventListener('click', () => {
        initAudio();
        for (let i = 0; i < 20; i++) {
            const x = Math.random() * width;
            const y = Math.random() * height;
            particles.push(new Particle(x, y, 'fake'));
        }
        if (particles.length > maxParticles * 1.5) {
            particles.splice(0, particles.length - Math.round(maxParticles * 1.5));
        }
        triggerShockwave(Math.random() * width, Math.random() * height, '#F43F5E');
        playBurstSound();
    });
}

if (pulseBtn) {
    pulseBtn.addEventListener('click', () => {
        initAudio();
        // Emits high verification wave from center and purifies nearby fake news
        triggerShockwave(width / 2, height / 2, '#10B981');
        particles.forEach(p => {
            const distFromCenter = Math.hypot(p.x - width / 2, p.y - height / 2);
            if (distFromCenter < 280) {
                p.type = 'real';
            }
        });
        playPulseSound();
    });
}

// Launch Engine
resizeCanvas();
initParticles();
setupEventListeners();
requestAnimationFrame(animate);
