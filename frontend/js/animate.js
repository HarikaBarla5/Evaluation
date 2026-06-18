// Animate signal fills on load
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".signal-fill[data-fill]").forEach((el) => {
    const target = el.getAttribute("data-fill");
    el.style.setProperty("--w", "0%");
    requestAnimationFrame(() => {
      setTimeout(() => {
        el.style.setProperty("--w", target + "%");
      }, 150);
    });
  });
});