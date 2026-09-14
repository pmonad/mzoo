/* Light/dark theme toggle. The palette lives in stylesheets/theme.css and is
 * driven by html[data-theme]; this script resolves the initial theme (stored
 * choice, else the system preference), keeps the choice in localStorage, and
 * swaps the highlight.js stylesheet so code blocks match the active theme. */
(function () {
  var KEY = "mzoo-theme";

  function systemDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function stored() {
    try {
      return localStorage.getItem(KEY);
    } catch (e) {
      return null;
    }
  }

  function hljsLink() {
    var links = document.querySelectorAll('link[href*="highlight.js"][href*="/styles/"]');
    return links.length ? links[links.length - 1] : null;
  }

  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    var link = hljsLink();
    if (link) {
      link.setAttribute(
        "href",
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/" +
          (theme === "dark" ? "github-dark.min.css" : "github.min.css")
      );
    }
    var btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.textContent = theme === "dark" ? "\u2600" : "\u263E"; // sun in dark, moon in light
      btn.setAttribute("aria-label", "Switch to " + (theme === "dark" ? "light" : "dark") + " theme");
      btn.title = btn.getAttribute("aria-label");
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    var initial = stored();
    if (initial !== "light" && initial !== "dark") {
      initial = systemDark() ? "dark" : "light";
    }

    var btn = document.createElement("button");
    btn.id = "theme-toggle";
    btn.type = "button";
    btn.addEventListener("click", function () {
      var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(KEY, next);
      } catch (e) {}
      apply(next);
    });
    document.body.appendChild(btn);

    apply(initial);

    // follow system changes while the user has not made an explicit choice
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
        if (!stored()) apply(e.matches ? "dark" : "light");
      });
    }

    // printing always uses the light palette: the code-highlight stylesheet
    // carries its own colors, so it has to be swapped alongside the CSS
    window.addEventListener("beforeprint", function () {
      apply("light");
    });
    window.addEventListener("afterprint", function () {
      apply(stored() || (systemDark() ? "dark" : "light"));
    });
  });
})();
