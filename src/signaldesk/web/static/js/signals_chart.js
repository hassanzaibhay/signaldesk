/**
 * Draw the signals scatter from the figure spec the server rendered.
 *
 * The figure is built in Python and shipped as JSON in a `json_script` block, so
 * this file decides nothing about what is plotted - it only puts the spec on the
 * page. That keeps the caption stating the point cap and the active filter in
 * one place, next to the query that produced them.
 *
 * Redrawn after every HTMX settle because the results fragment, chart div and
 * figure payload are all replaced together on a filter, a sort or a page change.
 */
(function () {
  "use strict";

  function draw() {
    var target = document.getElementById("signal-chart");
    var payload = document.getElementById("signal-figure");
    if (!target || !payload || typeof Plotly === "undefined") {
      // No chart on this render. An empty result set deliberately replaces the
      // plot with an explanation rather than drawing an empty pair of axes.
      return;
    }
    var figure = JSON.parse(payload.textContent);
    Plotly.newPlot(target, figure.data, figure.layout, figure.config);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", draw);
  } else {
    draw();
  }
  document.body.addEventListener("htmx:afterSettle", draw);
})();
