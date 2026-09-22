/* DocArchives dashboard charts (Chart.js, read-only presentation).
   Data comes from JSON in canvas data-* attributes rendered server-side.
   No framework, no fetching, no writes. */
(function () {
  'use strict';

  function ready(fn) {
    if (document.readyState !== 'loading') { fn(); }
    else { document.addEventListener('DOMContentLoaded', fn); }
  }

  ready(function () {
    if (typeof Chart === 'undefined') { return; }
    var reduceMotion = window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    var appFont = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
    Chart.defaults.font.family = appFont;
    Chart.defaults.font.size = 11;
    Chart.defaults.color = '#64748b';

    function tooltipBase() {
      return {
        backgroundColor: '#1f2b37',
        titleFont: { size: 11, weight: '600' },
        bodyFont: { size: 11 },
        padding: 8,
        cornerRadius: 8,
        displayColors: false
      };
    }

    function parseArray(raw, asInt) {
      try {
        var arr = JSON.parse(raw || '[]');
        if (!Array.isArray(arr)) { return []; }
        return arr.map(function (x) { return asInt ? (parseInt(x, 10) || 0) : String(x); });
      } catch (e) { return []; }
    }

    /* Center-total plugin for donuts. */
    var centerTotal = {
      id: 'docCenterTotal',
      afterDraw: function (chart, args, opts) {
        if (!opts || opts.text === undefined) { return; }
        var meta = chart.getDatasetMeta(0);
        if (!meta || !meta.data || !meta.data[0]) { return; }
        var ctx = chart.ctx;
        var x = meta.data[0].x, y = meta.data[0].y;
        ctx.save();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = '800 22px ' + appFont;
        ctx.fillStyle = '#1f2b37';
        ctx.fillText(String(opts.text), x, y - 8);
        ctx.font = '600 9px ' + appFont;
        ctx.fillStyle = '#64748b';
        ctx.fillText(opts.sub || '', x, y + 14);
        ctx.restore();
      }
    };
    Chart.register(centerTotal);

    /* Line/activity charts. */
    document.querySelectorAll('canvas[data-chart="activity"]').forEach(function (el) {
      var labels = parseArray(el.getAttribute('data-labels'), false);
      var counts = parseArray(el.getAttribute('data-counts'), true);
      if (!labels.length) { return; }
      new Chart(el, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: 'Vérifications',
            data: counts,
            borderColor: '#c1272d',
            backgroundColor: 'rgba(193, 39, 45, 0.04)',
            fill: true,
            tension: 0.25,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: '#c1272d',
            pointHoverBorderColor: '#fff'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: reduceMotion ? false : { duration: 400 },
          plugins: {
            legend: { display: false },
            tooltip: Object.assign(tooltipBase(), {
              callbacks: {
                label: function (c) {
                  var n = c.parsed.y;
                  return n + ' vérification' + (n > 1 ? 's' : '');
                }
              }
            })
          },
          scales: {
            x: { grid: { display: false }, ticks: { maxTicksLimit: 7, color: '#8a97a8' } },
            y: {
              beginAtZero: true,
              ticks: { precision: 0, maxTicksLimit: 4, color: '#8a97a8' },
              grid: { color: 'rgba(100, 116, 139, 0.08)' },
              border: { display: false }
            }
          }
        }
      });
    });

    /* Decision donuts. */
    document.querySelectorAll('canvas[data-chart="donut"]').forEach(function (el) {
      var values = [
        parseInt(el.getAttribute('data-conforme') || '0', 10) || 0,
        parseInt(el.getAttribute('data-manuel') || '0', 10) || 0,
        parseInt(el.getAttribute('data-nonconforme') || '0', 10) || 0
      ];
      var total = values[0] + values[1] + values[2];
      if (total <= 0) { return; }
      new Chart(el, {
        type: 'doughnut',
        data: {
          labels: ['Conforme', 'Contrôle manuel', 'Non conforme'],
          datasets: [{
            data: values,
            backgroundColor: ['#16a34a', '#d97706', '#c0362c'],
            borderColor: '#ffffff',
            borderWidth: 2,
            hoverOffset: 3
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '72%',
          animation: reduceMotion ? false : { duration: 400 },
          plugins: {
            legend: { display: false },
            tooltip: tooltipBase(),
            docCenterTotal: { text: total, sub: 'vérifications' }
          }
        }
      });
    });
  });
})();
