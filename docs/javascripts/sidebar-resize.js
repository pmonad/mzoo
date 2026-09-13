document.addEventListener("DOMContentLoaded", function () {
  var MIN_WIDTH = 180;
  var MAX_WIDTH = 700;
  var STORAGE_KEY = "mzoo-sidebar-width";

  var resizer = document.createElement("div");
  resizer.id = "sidebar-resizer";
  document.body.appendChild(resizer);

  function applyWidth(w) {
    document.documentElement.style.setProperty("--sidebar-width", w + "px");
    resizer.style.left = w + "px";
  }

  var saved = parseInt(localStorage.getItem(STORAGE_KEY), 10);
  var initial = isNaN(saved) ? 300 : saved;
  applyWidth(initial);

  var dragging = false;

  resizer.addEventListener("mousedown", function (e) {
    dragging = true;
    resizer.className = "dragging";
    document.body.style.userSelect = "none";
    e.preventDefault();
  });

  document.addEventListener("mousemove", function (e) {
    if (!dragging) return;
    var w = e.clientX;
    if (w < MIN_WIDTH) w = MIN_WIDTH;
    if (w > MAX_WIDTH) w = MAX_WIDTH;
    applyWidth(w);
  });

  document.addEventListener("mouseup", function () {
    if (!dragging) return;
    dragging = false;
    resizer.className = "";
    document.body.style.userSelect = "";
    var current = parseInt(resizer.style.left, 10);
    localStorage.setItem(STORAGE_KEY, current);
  });
});
