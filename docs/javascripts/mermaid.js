/* Mermaid styling driven by the RooAGI brand palette (https://rooagi.com).
   Keeps mermaid's "base" theme and supplies our own variables per colour scheme. */

const LINT_MERMAID_PALETTES = {
  default: {
    background: "#ffffff",
    primaryColor: "#eaf2f8",
    primaryTextColor: "#312a23",
    primaryBorderColor: "#316d98",
    secondaryColor: "#fbeef6",
    secondaryTextColor: "#312a23",
    secondaryBorderColor: "#984284",
    tertiaryColor: "#eef3ef",
    tertiaryTextColor: "#312a23",
    tertiaryBorderColor: "#4e715a",
    lineColor: "#6e665e",
    fontFamily: "Inter, sans-serif",
  },
  slate: {
    background: "#122e42",
    primaryColor: "#19415c",
    primaryTextColor: "#eef8ff",
    primaryBorderColor: "#6ba6d3",
    secondaryColor: "#421a38",
    secondaryTextColor: "#eef8ff",
    secondaryBorderColor: "#eda1d7",
    tertiaryColor: "#203025",
    tertiaryTextColor: "#eef8ff",
    tertiaryBorderColor: "#a9c7b1",
    lineColor: "#b5bfc7",
    fontFamily: "Inter, sans-serif",
  },
};

function lintMermaidScheme() {
  const scheme = document.body.getAttribute("data-md-color-scheme");
  return scheme === "slate" ? "slate" : "default";
}

function lintRenderMermaid() {
  const nodes = document.querySelectorAll(".mermaid");
  if (!nodes.length) {
    return;
  }
  nodes.forEach((node) => {
    if (!node.dataset.lintSource) {
      node.dataset.lintSource = node.textContent;
    } else {
      node.textContent = node.dataset.lintSource;
    }
    node.removeAttribute("data-processed");
  });
  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    securityLevel: "strict",
    themeVariables: LINT_MERMAID_PALETTES[lintMermaidScheme()],
  });
  mermaid.run({ nodes });
}

document$.subscribe(lintRenderMermaid);

if (typeof window.matchMedia === "function") {
  new MutationObserver(lintRenderMermaid).observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
}
