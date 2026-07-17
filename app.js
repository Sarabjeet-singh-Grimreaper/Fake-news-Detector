// Canvas & Context setup
const canvas = document.getElementById('simulation-canvas');
const ctx = canvas.getContext('2d');

// UI DOM elements
const countRealEl = document.getElementById('count-real');
const countFakeEl = document.getElementById('count-fake');
const countUnverifiedEl = document.getElementById('count-unverified');
const interactionSelect = document.getElementById('control-interaction');
const densitySlider = document.getElementById('control-density');
const densityVal = document.getElementById('density-val');
const speedSlider = document.getElementById('control-speed');
const speedVal = document.getElementById('speed-val');
const resetBtn = document.getElementById('btn-reset');
const injectBtn = document.getElementById('btn-inject');

// Simulation State
let particles = [];
let maxParticles = parseInt(densitySlider.value);
let globalSpeedMultiplier = parseFloat(speedSlider.value);
let dpr = window.devicePixelRatio || 1;
let width = window.innerWidth;
let height = window.innerHeight;

// Pointer state (mouse or touch)
let pointer = {
    x: 0,
    y: 0,
    active: false,
    radius: 120 // Interaction radius
};

// Colors matching CSS theme
const COLORS = {
    real: 'rgba(52, 211, 153, 0.85)',       // Emerald Green
    realTail: 'rgba(52, 211, 153, 0.15)',
    fake: 'rgba(248, 113, 113, 0.85)',       // Rose Red
    fakeTail: 'rgba(248, 113, 113, 0.15)',
    unverified: 'rgba(251, 191, 36, 0.75)',  // Amber
    unverifiedTail: 'rgba(251, 191, 36, 0.12)'
};

// Particle Class Definition
class Particle {
    constructor(x, y, type = null) {
        this.x = x !== undefined ? x : Math.random() * width;
        this.y = y !== undefined ? y : Math.random() * height;
        
        // Random velocities
        const speed = (Math.random() * 1.5 + 0.5);
        const angle = Math.random() * Math.PI * 2;
        this.vx = Math.cos(angle) * speed;
        this.vy = Math.sin(angle) * speed;
        
        // Define particle classification
        if (type) {
            this.type = type;
        } else {
            const rand = Math.random();
            if (rand < 0.35) this.type = 'real';
            else if (rand < 0.65) this.type = 'fake';
            else this.type = 'unverified';
        }
        
        // Set size and mass based on type
        this.radius = Math.random() * 3 + 3;
        if (this.type === 'fake') {
            this.radius += 1.5; // Misinformation particles are slightly bloated
        }
        this.mass = this.radius;
    }

    update() {
        // Apply physics
        this.x += this.vx * globalSpeedMultiplier;
        this.y += this.vy * globalSpeedMultiplier;

        // Friction / drag
        this.vx *= 0.99;
        this.vy *= 0.99;

        // Bounce off canvas boundaries
        if (this.x - this.radius < 0) {
            this.x = this.radius;
            this.vx *= -1;
        } else if (this.x + this.radius > width) {
            this.x = width - this.radius;
            this.vx *= -1;
        }

        if (this.y - this.radius < 0) {
            this.y = this.radius;
            this.vy *= -1;
        } else if (this.y + this.radius > height) {
            this.y = height - this.radius;
            this.vy *= -1;
        }

        // Interact with pointer
        if (pointer.active) {
            const dx = this.x - pointer.x;
            const dy = this.y - pointer.y;
            const dist = Math.hypot(dx, dy);

            if (dist < pointer.radius) {
                const force = (pointer.radius - dist) / pointer.radius;
                const angle = Math.atan2(dy, dx);
                
                const currentMode = interactionSelect.value;
                if (currentMode === 'repel') {
                    // Push particles away
                    this.vx += Math.cos(angle) * force * 0.8;
                    this.vy += Math.sin(angle) * force * 0.8;
                } else if (currentMode === 'attract') {
                    // Pull particles towards the cursor
                    this.vx -= Math.cos(angle) * force * 0.8;
                    this.vy -= Math.sin(angle) * force * 0.8;
                }
            }
        }
    }

    draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[this.type];
        ctx.fill();

        // Draw outer glow/field for particles
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius + 4, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[this.type + 'Tail'];
        ctx.fill();
    }
}

// Set correct pixel ratio and dimensions
function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    width = window.innerWidth;
    height = window.innerHeight;

    // Logical dimensions
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    // High DPI Canvas Scaling
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
}

// Setup Simulation Grid Particles
function initParticles() {
    particles = [];
    for (let i = 0; i < maxParticles; i++) {
        particles.push(new Particle());
    }
}

// Map desktop & mobile event listeners
function setupEventListeners() {
    // Resize Listener
    window.addEventListener('resize', () => {
        resizeCanvas();
        // Adjust particle positions if outside boundaries
        particles.forEach(p => {
            if (p.x > width) p.x = Math.random() * width;
            if (p.y > height) p.y = Math.random() * height;
        });
    });

    // Pointer Event Listeners (Mouse & Touch merged natively)
    const updatePointer = (e) => {
        // Native pointer position translation
        pointer.x = e.clientX;
        pointer.y = e.clientY;
    };

    canvas.addEventListener('pointerdown', (e) => {
        pointer.active = true;
        updatePointer(e);
        
        // Spawn action trigger
        if (interactionSelect.value === 'spawn') {
            for (let i = 0; i < 5; i++) {
                particles.push(new Particle(pointer.x + (Math.random() - 0.5) * 20, pointer.y + (Math.random() - 0.5) * 20));
            }
            // Keep cap in check
            if (particles.length > maxParticles * 1.5) {
                particles.splice(0, particles.length - Math.round(maxParticles * 1.5));
            }
        }
    });

    canvas.addEventListener('pointermove', (e) => {
        updatePointer(e);
    });

    canvas.addEventListener('pointerup', () => { pointer.active = false; });
    canvas.addEventListener('pointercancel', () => { pointer.active = false; });
    canvas.addEventListener('pointerleave', () => { pointer.active = false; });

    // Prevent default touch interactions (zoom, double-tap, pan)
    canvas.addEventListener('touchstart', (e) => e.preventDefault(), { passive: false });
    canvas.addEventListener('touchmove', (e) => e.preventDefault(), { passive: false });
}

// Count dynamic stats and update UI elements
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
}

// Particle Collision & Infection Logic
function handleCollisions() {
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const p1 = particles[i];
            const p2 = particles[j];

            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const dist = Math.hypot(dx, dy);
            const minDist = p1.radius + p2.radius;

            if (dist < minDist) {
                // Perform clean particle-to-particle bouncing physics
                const angle = Math.atan2(dy, dx);
                const sin = Math.sin(angle);
                const cos = Math.cos(angle);

                // Rotate velocities
                const vx1 = p1.vx * cos + p1.vy * sin;
                const vy1 = p1.vy * cos - p1.vx * sin;
                const vx2 = p2.vx * cos + p2.vy * sin;
                const vy2 = p2.vy * cos - p2.vx * sin;

                // Resolve velocities after collision (elastic collision)
                const vx1Final = ((p1.mass - p2.mass) * vx1 + 2 * p2.mass * vx2) / (p1.mass + p2.mass);
                const vx2Final = ((p2.mass - p1.mass) * vx2 + 2 * p1.mass * vx1) / (p1.mass + p2.mass);

                // Update velocities back
                p1.vx = vx1Final * cos - vy1 * sin;
                p1.vy = vy1 * cos + vx1Final * sin;
                p2.vx = vx2Final * cos - vy2 * sin;
                p2.vy = vy2 * cos + vx2Final * sin;

                // Prevent sticking by separating particles
                const overlap = minDist - dist;
                const sepX = Math.cos(angle) * (overlap / 2);
                const sepY = Math.sin(angle) * (overlap / 2);
                p1.x -= sepX;
                p1.y -= sepY;
                p2.x += sepX;
                p2.y += sepY;

                // Interactive dynamics: Infection / Preprocessing
                // If a "Fake" article collides with "Unverified", it has a chance to contaminate it
                if (p1.type === 'fake' && p2.type === 'unverified' && Math.random() < 0.15) {
                    p2.type = 'fake';
                } else if (p2.type === 'fake' && p1.type === 'unverified' && Math.random() < 0.15) {
                    p1.type = 'fake';
                }
                
                // If a "Real" verified article collides with "Unverified", it has a chance to verify it
                if (p1.type === 'real' && p2.type === 'unverified' && Math.random() < 0.20) {
                    p2.type = 'real';
                } else if (p2.type === 'real' && p1.type === 'unverified' && Math.random() < 0.20) {
                    p1.type = 'real';
                }
            }
        }
    }
}

// Main Draw Loop
function animate() {
    ctx.clearRect(0, 0, width, height);

    // Apply collision and update physics
    handleCollisions();

    particles.forEach(p => {
        p.update();
        p.draw();
    });

    // Draw interaction ripple ring
    if (pointer.active && interactionSelect.value !== 'spawn') {
        ctx.beginPath();
        ctx.arc(pointer.x, pointer.y, pointer.radius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.12)';
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(pointer.x, pointer.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(56, 189, 248, 0.4)';
        ctx.fill();
    }

    updateStats();
    requestAnimationFrame(animate);
}

// Connect UI controls
densitySlider.addEventListener('input', (e) => {
    maxParticles = parseInt(e.target.value);
    densityVal.textContent = maxParticles;
    
    // Smoothly scale particle array size
    if (particles.length < maxParticles) {
        while (particles.length < maxParticles) {
            particles.push(new Particle());
        }
    } else {
        particles.splice(maxParticles);
    }
});

speedSlider.addEventListener('input', (e) => {
    globalSpeedMultiplier = parseFloat(e.target.value);
    speedVal.textContent = `${globalSpeedMultiplier.toFixed(1)}x`;
});

resetBtn.addEventListener('click', () => {
    initParticles();
});

injectBtn.addEventListener('click', () => {
    // Inject a massive burst of Fake news particles at random spots
    for (let i = 0; i < 15; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        particles.push(new Particle(x, y, 'fake'));
    }
    // Remove oldest to stay reasonably performant
    if (particles.length > maxParticles * 1.5) {
        particles.splice(0, particles.length - Math.round(maxParticles * 1.5));
    }
});

// Run
resizeCanvas();
initParticles();
setupEventListeners();
animate();
