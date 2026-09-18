(function () {
  function initSlideshows() {
    document.querySelectorAll('[data-dashboard-slideshow]').forEach(function (root) {
      if (root.dataset.ready) return;
      root.dataset.ready = 'true';
      var slides = Array.from(root.querySelectorAll('[data-dashboard-slide]'));
      var dots = Array.from(root.querySelectorAll('[data-dashboard-dot]'));
      var track = root.querySelector('.dashboard-slideshow__track');
      var current = 0;
      function show(index) {
        current = (index + slides.length) % slides.length;
        track.style.transform = 'translateX(-' + (current * 100) + '%)';
        slides.forEach(function (slide, i) { slide.classList.toggle('is-active', i === current); });
        dots.forEach(function (dot, i) {
          var active = i === current;
          dot.classList.toggle('is-active', active);
          dot.setAttribute('aria-selected', String(active));
        });
      }
      root.querySelector('[data-dashboard-prev]').addEventListener('click', function () { show(current - 1); });
      root.querySelector('[data-dashboard-next]').addEventListener('click', function () { show(current + 1); });
      dots.forEach(function (dot) { dot.addEventListener('click', function () { show(Number(dot.dataset.dashboardDot)); }); });
      root.addEventListener('keydown', function (event) {
        if (event.key === 'ArrowLeft') show(current - 1);
        if (event.key === 'ArrowRight') show(current + 1);
      });
    });
  }
  document.addEventListener('DOMContentLoaded', initSlideshows);
  if (window.document$) document$.subscribe(initSlideshows);
})();
