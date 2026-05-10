/* ===== Particle Background ===== */
const canvas = document.getElementById('particles');
const ctx = canvas.getContext('2d');
let particles = [];
let mouse = { x: null, y: null };

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);
document.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });

class Particle {
  constructor() {
    this.x = Math.random() * canvas.width;
    this.y = Math.random() * canvas.height;
    this.size = Math.random() * 2 + 0.5;
    this.speedX = (Math.random() - 0.5) * 0.5;
    this.speedY = (Math.random() - 0.5) * 0.5;
    this.opacity = Math.random() * 0.5 + 0.1;
  }
  update() {
    this.x += this.speedX;
    this.y += this.speedY;
    if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
    if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
  }
  draw() {
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(99, 102, 241, ${this.opacity})`;
    ctx.fill();
  }
}

function initParticles() {
  const count = Math.min(80, Math.floor((canvas.width * canvas.height) / 15000));
  particles = [];
  for (let i = 0; i < count; i++) particles.push(new Particle());
}
initParticles();
window.addEventListener('resize', initParticles);

function connectParticles() {
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      const dx = particles[i].x - particles[j].x;
      const dy = particles[i].y - particles[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 120) {
        ctx.beginPath();
        ctx.strokeStyle = `rgba(99, 102, 241, ${0.08 * (1 - dist / 120)})`;
        ctx.lineWidth = 0.5;
        ctx.moveTo(particles[i].x, particles[i].y);
        ctx.lineTo(particles[j].x, particles[j].y);
        ctx.stroke();
      }
    }
  }
}

function animateParticles() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  particles.forEach(p => { p.update(); p.draw(); });
  connectParticles();
  requestAnimationFrame(animateParticles);
}
animateParticles();

/* ===== Navbar Scroll ===== */
const navbar = document.querySelector('.navbar');
const backTop = document.querySelector('.back-top');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 50);
  if (backTop) backTop.classList.toggle('visible', window.scrollY > 500);
});

/* ===== Mobile Menu ===== */
const mobileToggle = document.querySelector('.mobile-toggle');
const navLinks = document.querySelector('.nav-links');
if (mobileToggle) {
  mobileToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
    const icon = mobileToggle.querySelector('.hamburger');
    if (icon) icon.classList.toggle('active');
  });
  navLinks.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => {
      navLinks.classList.remove('open');
      const icon = mobileToggle.querySelector('.hamburger');
      if (icon) icon.classList.remove('active');
    });
  });
}

/* ===== Active Nav Link ===== */
const sections = document.querySelectorAll('section[id]');
window.addEventListener('scroll', () => {
  const scrollY = window.scrollY + 150;
  sections.forEach(section => {
    const top = section.offsetTop;
    const height = section.offsetHeight;
    const id = section.getAttribute('id');
    const link = document.querySelector(`.nav-links a[href="#${id}"]`);
    if (link) {
      if (scrollY >= top && scrollY < top + height) link.classList.add('active');
      else link.classList.remove('active');
    }
  });
});

/* ===== Scroll Reveal ===== */
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
    }
  });
}, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

/* ===== Counter Animation ===== */
function animateCounter(el) {
  const target = el.getAttribute('data-target');
  const isDecimal = target.includes('.');
  const targetNum = parseFloat(target);
  const duration = 2000;
  const start = performance.now();
  const suffix = el.getAttribute('data-suffix') || '';
  const prefix = el.getAttribute('data-prefix') || '';

  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = eased * targetNum;
    el.textContent = prefix + (isDecimal ? current.toFixed(1) : Math.floor(current)) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting && !entry.target.dataset.counted) {
      entry.target.dataset.counted = 'true';
      animateCounter(entry.target);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('.stat-number').forEach(el => counterObserver.observe(el));

/* ===== Typing Effect ===== */
const typingEl = document.querySelector('.typing-text');
if (typingEl) {
  const words = ['Helmets', 'Safety Vests', 'Masks', 'Gloves', 'Safety Cones'];
  let wordIndex = 0, charIndex = 0, isDeleting = false;

  function typeEffect() {
    const current = words[wordIndex];
    typingEl.textContent = current.substring(0, charIndex);

    if (!isDeleting && charIndex < current.length) {
      charIndex++;
      setTimeout(typeEffect, 80);
    } else if (isDeleting && charIndex > 0) {
      charIndex--;
      setTimeout(typeEffect, 40);
    } else if (!isDeleting) {
      isDeleting = true;
      setTimeout(typeEffect, 1500);
    } else {
      isDeleting = false;
      wordIndex = (wordIndex + 1) % words.length;
      setTimeout(typeEffect, 300);
    }
  }
  setTimeout(typeEffect, 1000);
}

/* ===== Back to Top ===== */
if (backTop) {
  backTop.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

/* ===== Dashboard Live Simulation ===== */
function simulateDashboard() {
  const alertLog = document.querySelector('.alert-log-items');
  if (!alertLog) return;
  const alerts = [
    { text: 'Helmet detected — Zone A', type: 'g' },
    { text: 'No mask warning — Zone B', type: 'r' },
    { text: 'Safety vest confirmed', type: 'g' },
    { text: 'New worker entered — Zone C', type: 'y' },
    { text: 'PPE compliance check passed', type: 'g' },
    { text: 'Gloves not detected — Zone A', type: 'r' },
    { text: 'All PPE verified — Zone D', type: 'g' },
  ];

  setInterval(() => {
    const alert = alerts[Math.floor(Math.random() * alerts.length)];
    const item = document.createElement('div');
    item.className = 'alert-item';
    item.innerHTML = `<span class="alert-dot ${alert.type}"></span><span>${alert.text}</span>`;
    alertLog.prepend(item);
    if (alertLog.children.length > 5) alertLog.removeChild(alertLog.lastChild);
  }, 3000);
}
simulateDashboard();
