/* ===== SAFE3D-AI — Video Carousel ===== */
(function () {
  const track = document.querySelector('.carousel-track');
  const slides = track ? track.querySelectorAll('.carousel-slide') : [];
  const dotsContainer = document.querySelector('.carousel-pagination');
  const prevBtn = document.querySelector('.carousel-btn--prev');
  const nextBtn = document.querySelector('.carousel-btn--next');

  if (!slides.length) return;

  let current = 0;
  let autoplayInterval;
  const AUTOPLAY_DELAY = 4500;

  // Build dots
  slides.forEach((_, i) => {
    const dot = document.createElement('button');
    dot.className = 'carousel-dot' + (i === 0 ? ' active' : '');
    dot.setAttribute('aria-label', `Go to slide ${i + 1}`);
    dot.addEventListener('click', () => goTo(i));
    dotsContainer.appendChild(dot);
  });

  const dots = dotsContainer.querySelectorAll('.carousel-dot');

  function goTo(index) {
    current = (index + slides.length) % slides.length;
    track.style.transform = `translateX(-${current * 100}%)`;
    dots.forEach((d, i) => d.classList.toggle('active', i === current));
    resetAutoplay();
  }

  function next() { goTo(current + 1); }
  function prev() { goTo(current - 1); }

  if (prevBtn) prevBtn.addEventListener('click', prev);
  if (nextBtn) nextBtn.addEventListener('click', next);

  // Keyboard navigation
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') prev();
    if (e.key === 'ArrowRight') next();
  });

  // Touch / swipe support
  let touchStartX = 0;
  const container = document.querySelector('.carousel-track-container');
  if (container) {
    container.addEventListener('touchstart', (e) => { touchStartX = e.changedTouches[0].screenX; }, { passive: true });
    container.addEventListener('touchend', (e) => {
      const diff = e.changedTouches[0].screenX - touchStartX;
      if (Math.abs(diff) > 40) { diff > 0 ? prev() : next(); }
    }, { passive: true });
  }

  // Autoplay
  function startAutoplay() { autoplayInterval = setInterval(next, AUTOPLAY_DELAY); }
  function stopAutoplay() { clearInterval(autoplayInterval); }
  function resetAutoplay() { stopAutoplay(); startAutoplay(); }

  startAutoplay();

  // Pause on hover / focus
  const wrapper = document.querySelector('.carousel-wrapper');
  if (wrapper) {
    wrapper.addEventListener('mouseenter', stopAutoplay);
    wrapper.addEventListener('mouseleave', startAutoplay);
    wrapper.addEventListener('focusin', stopAutoplay);
    wrapper.addEventListener('focusout', startAutoplay);
  }
})();

// Nav shadow on scroll
(function () {
  const nav = document.querySelector('nav');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 10);
  }, { passive: true });
})();
