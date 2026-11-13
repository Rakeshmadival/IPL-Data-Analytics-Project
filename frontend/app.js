const pageNames = {
  overview: "IPL Analytics Dashboard",
  teams: "Team analysis",
  insights: "Match insights",
  powerbi: "Power BI reports",
  prediction: "Prediction lab",
  model: "Model evaluation"
};

let modelConnected = false;

function showPage(id, updateHistory = true) {
  if (!Object.hasOwn(pageNames, id)) id = "overview";
  document.querySelectorAll("[data-page]").forEach((page) => {
    page.classList.toggle("page-hidden", page.dataset.page !== id);
  });
  document.querySelectorAll("nav a").forEach((link) => {
    const active = link.hash === `#${id}`;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });

  const heading = document.querySelector("header h1");
  const pagePill = document.getElementById("page-pill");
  heading.textContent = pageNames[id];
  pagePill.textContent = id === "powerbi"
    ? "● Power BI report"
    : id === "prediction"
      ? "● Experimental model"
      : "● Historical data";

  if (updateHistory && location.hash !== `#${id}`) {
    history.pushState({ page: id }, "", `#${id}`);
  }
  window.scrollTo({ top: 0, behavior: "instant" });
}

document.querySelectorAll('a[href^="#"]').forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    showPage(link.hash.slice(1));
  });
});

window.addEventListener("popstate", () => {
  showPage(location.hash.slice(1) || "overview", false);
});

showPage(location.hash.slice(1) || "overview", false);

function formatNumber(value) {
  return Number(value).toLocaleString("en-US");
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "—";
}

async function loadMetrics() {
  const overviewStatus = document.getElementById("overview-status");
  const evaluationStatus = document.getElementById("evaluation-status");

  try {
    const response = await fetch("/api/metrics");
    if (!response.ok) throw new Error(`Metrics API returned ${response.status}`);
    const metrics = await response.json();
    const accuracyText = formatPercent(metrics.random_forest_accuracy);
    const baselineText = formatPercent(metrics.toss_winner_baseline_accuracy);
    const accuracy = Number.isFinite(metrics.random_forest_accuracy)
      ? metrics.random_forest_accuracy * 100
      : 0;
    const baseline = Number.isFinite(metrics.toss_winner_baseline_accuracy)
      ? metrics.toss_winner_baseline_accuracy * 100
      : 0;

    document.getElementById("hero-matches").textContent = metrics.matches == null
      ? "—"
      : formatNumber(metrics.matches);
    document.getElementById("hero-period").textContent = metrics.dataset_period || "Dataset unavailable";
    document.getElementById("overview-period").textContent = metrics.dataset_period
      ? `Historical dataset · ${metrics.dataset_period}`
      : "Historical dataset unavailable";
    document.getElementById("matches-count").textContent = metrics.matches == null
      ? "—"
      : formatNumber(metrics.matches);
    document.getElementById("deliveries-count").textContent = metrics.deliveries == null
      ? "—"
      : formatNumber(metrics.deliveries);
    document.getElementById("overview-accuracy").textContent = accuracyText;
    document.getElementById("overview-baseline").textContent = baselineText;

    document.getElementById("prediction-accuracy").textContent = accuracyText;
    document.getElementById("prediction-baseline").textContent = baselineText;
    document.getElementById("prediction-test-size").textContent = metrics.test_matches == null
      ? "—"
      : formatNumber(metrics.test_matches);

    document.getElementById("eval-accuracy").textContent = accuracyText;
    document.getElementById("eval-baseline").textContent = baselineText;
    document.getElementById("eval-test-size").textContent = metrics.test_matches == null
      ? "—"
      : formatNumber(metrics.test_matches);
    document.getElementById("eval-model-bar").style.width = `${accuracy}%`;
    document.getElementById("eval-baseline-bar").style.width = `${baseline}%`;
    document.getElementById("eval-model-label").textContent = accuracyText;
    document.getElementById("eval-baseline-label").textContent = baselineText;

    if (metrics.evaluation_available) {
      const rolling = metrics.rolling_origin_matches
        ? ` Rolling-origin evaluation: ${formatPercent(metrics.rolling_origin_accuracy)} vs ${formatPercent(metrics.rolling_origin_toss_baseline_accuracy)} toss baseline across ${formatNumber(metrics.rolling_origin_matches)} matches.`
        : "";
      const interval = metrics.random_forest_accuracy_95_ci
        ? ` 95% Wilson interval: ${(metrics.random_forest_accuracy_95_ci[0] * 100).toFixed(1)}–${(metrics.random_forest_accuracy_95_ci[1] * 100).toFixed(1)}%.`
        : "";
      evaluationStatus.textContent = `Chronological holdout: ${formatNumber(metrics.test_matches)} matches.${interval}${rolling}`;
      const limitations = [
        `On the ${formatNumber(metrics.test_matches)}-match chronological holdout, the model scored ${accuracyText} versus ${baselineText} for the toss-winner baseline.`,
        metrics.matches_with_unseen_team
          ? `${formatNumber(metrics.matches_with_unseen_team)} holdout matches included a team absent from prior training data; ${formatNumber(metrics.matches_with_unseen_winner)} winners were not seen in training.`
          : "",
        metrics.rolling_origin_matches
          ? `Across rolling-origin checks, the model scored ${formatPercent(metrics.rolling_origin_accuracy)} versus ${formatPercent(metrics.rolling_origin_toss_baseline_accuracy)} for the toss baseline on ${formatNumber(metrics.rolling_origin_matches)} later-season matches.`
          : "",
        "The model score is not a calibrated probability. Its selectable categories are limited to the data it was trained on.",
        `Training data ends ${metrics.model_training_end_date || "at the latest supplied season"}.`,
        `Dataset provenance and license: ${(metrics.data_provenance || "not provided").replace(/\.$/, "")}.`
      ].filter(Boolean);
      document.getElementById("evaluation-limitations").textContent = limitations.join(" ");
    } else {
      evaluationStatus.textContent = metrics.evaluation_error
        ? `Reproducible model evaluation is unavailable: ${metrics.evaluation_error}`
        : "Reproducible model evaluation is unavailable. Run python backend\\analytics.py with the supplied matches.csv.";
      document.getElementById("evaluation-limitations").textContent =
        "The model evaluation report is unavailable. No accuracy or baseline result is displayed until it has been generated from the supplied data.";
    }

    overviewStatus.textContent = metrics.dataset_period
      ? "Historical counts and model scores are loaded from the supplied data and evaluation report."
      : "Historical dataset unavailable. Configure the CSV file paths to show data.";
  } catch (error) {
    overviewStatus.textContent = `Historical metrics unavailable: ${error.message}`;
    evaluationStatus.textContent = `Evaluation metrics unavailable: ${error.message}`;
  }
}

function appendRanking(containerId, rows, maximum, className) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  rows.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = className;

    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = String(index + 1).padStart(2, "0");

    const name = document.createElement("span");
    name.className = "rank-name";
    name.textContent = item.label;
    if (item.detail) {
      const detail = document.createElement("small");
      detail.textContent = item.detail;
      name.append(detail);
    }

    const track = document.createElement("span");
    track.className = "rank-track";
    const fill = document.createElement("span");
    fill.className = "rank-fill";
    fill.style.width = `${maximum ? item.count / maximum * 100 : 0}%`;
    track.append(fill);

    const value = document.createElement("b");
    value.textContent = formatNumber(item.count);
    row.append(rank, name, track, value);
    container.append(row);
  });
}

function setResultCount(id, results, key) {
  document.getElementById(id).textContent = formatNumber(results[key] || 0);
}

async function loadAnalytics() {
  try {
    const response = await fetch("/api/analytics");
    const analytics = await response.json();
    if (!response.ok) {
      throw new Error(analytics.detail || analytics.error || `Analytics API returned ${response.status}`);
    }

    const rankedTeams = analytics.team_rankings;
    const teamRows = rankedTeams.map((team) => ({
      label: team.team,
      count: team.wins,
      detail: `${formatNumber(team.matches_played)} played · ${team.win_percentage.toFixed(2)}% wins`
    }));
    appendRanking(
      "team-rankings",
      teamRows,
      Math.max(...rankedTeams.map((team) => team.wins)),
      "team-rank"
    );
    const leadingTeam = rankedTeams[0];
    if (leadingTeam) {
      document.getElementById("leading-team-rate").textContent = `${leadingTeam.win_percentage.toFixed(2)}%`;
      document.getElementById("leading-team-name").textContent = leadingTeam.team;
      document.getElementById("leading-team-record").textContent =
        `${formatNumber(leadingTeam.wins)} wins from ${formatNumber(leadingTeam.matches_played)} matches played.`;
    }
    document.getElementById("team-status").textContent =
      `Computed from matches.csv; ${formatNumber(analytics.data_quality.missing_winner_matches)} no-result matches are included in appearances but not wins.`;

    const toss = analytics.toss;
    const tossTotal = toss.winner_wins + toss.winner_loses;
    document.getElementById("toss-win-rate").textContent =
      toss.win_percentage == null ? "—" : `${toss.win_percentage.toFixed(2)}%`;
    document.getElementById("toss-won-count").textContent = formatNumber(toss.winner_wins);
    document.getElementById("toss-lost-count").textContent = formatNumber(toss.winner_loses);
    document.getElementById("toss-won-bar").style.width =
      `${tossTotal ? toss.winner_wins / tossTotal * 100 : 0}%`;
    document.getElementById("toss-lost-bar").style.width =
      `${tossTotal ? toss.winner_loses / tossTotal * 100 : 0}%`;
    document.getElementById("toss-detail").textContent =
      `Toss decisions: field ${formatNumber(toss.decisions.field || 0)} · bat ${formatNumber(toss.decisions.bat || 0)} · no result ${formatNumber(toss.no_result)}.`;

    const results = analytics.results;
    const resultKeys = ["wickets", "runs", "tie", "no result"];
    const resultColors = ["#ff814f", "#5c8fbe", "#a48bd0", "#596675"];
    const total = analytics.matches;
    let cumulative = 0;
    const stops = resultKeys.map((key, index) => {
      const start = cumulative;
      cumulative += total ? (results[key] || 0) / total * 100 : 0;
      return `${resultColors[index]} ${start.toFixed(2)}% ${cumulative.toFixed(2)}%`;
    });
    document.getElementById("result-donut").style.background = `conic-gradient(${stops.join(", ")})`;
    document.getElementById("result-total").textContent = formatNumber(total);
    setResultCount("result-wickets", results, "wickets");
    setResultCount("result-runs", results, "runs");
    setResultCount("result-ties", results, "tie");
    setResultCount("result-no-result", results, "no result");
    document.getElementById("result-detail").textContent =
      `${formatNumber(analytics.completed_matches)} matches recorded a winner; ties include a recorded winner where present.`;

    const venues = analytics.top_venues.map((venue) => ({
      label: venue.venue,
      count: venue.matches,
      detail: "matches"
    }));
    appendRanking(
      "venue-rankings",
      venues,
      Math.max(...analytics.top_venues.map((venue) => venue.matches)),
      "venue-rank"
    );
    document.getElementById("venue-status").textContent =
      `Top venues by match count. ${formatNumber(analytics.data_quality.duplicate_match_ids)} duplicate match IDs found.`;

    const recentSeasons = analytics.season_matches.slice(-6).map((season) => ({
      label: season.season,
      count: season.matches,
      detail: "matches"
    }));
    appendRanking(
      "season-rankings",
      recentSeasons,
      Math.max(...recentSeasons.map((season) => season.matches)),
      "season-rank"
    );
    document.getElementById("historical-context").textContent =
      `${formatNumber(analytics.matches)} matches across ${analytics.dataset_period}; ` +
      `${analytics.deliveries == null ? "delivery file not configured" : `${formatNumber(analytics.deliveries)} deliveries checked`}. ` +
      `Dataset provenance and license: ${analytics.data_quality.provenance}`;
  } catch (error) {
    document.getElementById("team-status").textContent = `Team data unavailable: ${error.message}`;
    document.getElementById("toss-detail").textContent = `Toss data unavailable: ${error.message}`;
    document.getElementById("result-detail").textContent = `Match results unavailable: ${error.message}`;
    document.getElementById("venue-status").textContent = `Venue data unavailable: ${error.message}`;
    document.getElementById("historical-context").textContent = `Historical data unavailable: ${error.message}`;
  }
}

function fillSelect(id, values) {
  const select = document.getElementById(id);
  select.replaceChildren();
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  });
}

function compareSeasons(first, second) {
  const firstYear = Number.parseInt(first, 10);
  const secondYear = Number.parseInt(second, 10);
  return firstYear - secondYear || first.localeCompare(second);
}

function setPredictionControls(enabled) {
  ["season", "a", "b", "venue", "toss", "decision", "batfirst", "predict"]
    .forEach((id) => {
      document.getElementById(id).disabled = !enabled;
    });
}

async function loadModelOptions() {
  const message = document.getElementById("msg");
  const result = document.getElementById("result");
  setPredictionControls(false);

  try {
    const healthResponse = await fetch("/api/health");
    if (!healthResponse.ok) throw new Error(`Health API returned ${healthResponse.status}`);
    const health = await healthResponse.json();
    if (!health.model_loaded || !health.model_current) {
      throw new Error(health.model_error || "The trained model is not loaded or is out of date.");
    }

    const optionsResponse = await fetch("/api/options");
    const options = await optionsResponse.json();
    if (!optionsResponse.ok) {
      throw new Error(options.detail || options.error || `Options API returned ${optionsResponse.status}`);
    }

    fillSelect("season", [...options.seasons_seen_in_training].sort(compareSeasons));
    fillSelect("a", options.teams_supported);
    fillSelect("b", options.teams_supported);
    fillSelect("venue", options.venues_seen_in_training);
    fillSelect("decision", options.toss_decisions);

    modelConnected = true;
    setPredictionControls(true);
    message.textContent = "Connected to the trained model. Options are limited to categories seen in training.";
  } catch (error) {
    modelConnected = false;
    result.hidden = true;
    message.textContent = `Prediction unavailable: ${error.message}`;
  }
}

const predictionForm = document.getElementById("prediction-form");
predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = document.getElementById("msg");
  const result = document.getElementById("result");
  const button = document.getElementById("predict");
  result.hidden = true;

  if (!modelConnected) {
    message.textContent = "Prediction unavailable because the trained model is not connected.";
    return;
  }

  const team1 = document.getElementById("a").value;
  const team2 = document.getElementById("b").value;
  if (team1 === team2) {
    message.textContent = "Choose two different teams.";
    return;
  }

  message.textContent = "Running the trained Random Forest pipeline…";
  button.disabled = true;
  const payload = {
    season: document.getElementById("season").value,
    team1,
    team2,
    venue: document.getElementById("venue").value,
    toss_winner_is_team1: document.getElementById("toss").value,
    toss_decision: document.getElementById("decision").value,
    team1_batting_first: document.getElementById("batfirst").value
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const prediction = await response.json();
    if (!response.ok) {
      throw new Error(prediction.message || prediction.detail || `Prediction API returned ${response.status}`);
    }

    const title = document.createElement("strong");
    title.textContent = prediction.predicted_winner;
    const score = document.createElement("span");
    score.textContent = `Model score: ${prediction.model_score}% (not calibrated)`;
    const notice = document.createElement("p");
    notice.textContent = prediction.notice;
    result.replaceChildren(title, score, notice);
    result.hidden = false;
    message.textContent = "Prediction generated by the connected trained model.";
  } catch (error) {
    message.textContent = error.message;
  } finally {
    button.disabled = !modelConnected;
  }
});

loadMetrics();
loadAnalytics();
loadModelOptions();
