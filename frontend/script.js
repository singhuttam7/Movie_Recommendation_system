/* =========================================================
   CONFIGURATION
========================================================= */

const API_BASE =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://movie-recommendation-system-2-ydx0.onrender.com";

// When deployed, replace with your backend URL.
// Example:
// const API_BASE = "https://your-backend.onrender.com";

const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500";

/* =========================================================
   DOM ELEMENTS
========================================================= */

const views = document.querySelectorAll(".view");

const navItems = document.querySelectorAll(".nav-item");

const categoryLinks = document.querySelectorAll(".category-link");

const sidebar = document.querySelector(".sidebar");

const sidebarOverlay = document.getElementById("sidebarOverlay");

const mobileMenuButton = document.getElementById("mobileMenuButton");

const homeMoviesGrid = document.getElementById("homeMoviesGrid");

const searchMoviesGrid = document.getElementById("searchMoviesGrid");

const movieDetailsContainer = document.getElementById("movieDetailsContainer");

const tfidfRecommendations = document.getElementById("tfidfRecommendations");

const genreRecommendations = document.getElementById("genreRecommendations");

const homeSearchInput = document.getElementById("homeSearchInput");

const movieSearchInput = document.getElementById("movieSearchInput");

const homeSearchButton = document.getElementById("homeSearchButton");

const suggestionsContainer = document.getElementById("suggestionsContainer");

const suggestionsList = document.getElementById("suggestionsList");

const resultsCount = document.getElementById("resultsCount");

const aiMovieInput = document.getElementById("aiMovieInput");

const aiAnalyzeButton = document.getElementById("aiAnalyzeButton");

const aiResultContainer = document.getElementById("aiResultContainer");

const aiResult = document.getElementById("aiResult");

const loadingOverlay = document.getElementById("loadingOverlay");

const loadingText = document.getElementById("loadingText");

const toast = document.getElementById("toast");

const toastMessage = document.getElementById("toastMessage");

const statusIndicator = document.getElementById("statusIndicator");

const apiStatusText = document.getElementById("apiStatusText");

/* =========================================================
   APPLICATION STATE
========================================================= */

let currentCategory = "trending";

let searchTimeout = null;

let searchController = null;

let previousView = "search";

let toastTimeout = null;

/* =========================================================
   API FUNCTIONS
========================================================= */

async function apiGet(path, params = {}, signal = undefined) {
  const url = new URL(API_BASE + path);

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.append(key, value);
    }
  });

  try {
    const response = await fetch(url, {
      signal,
    });

    if (!response.ok) {
      let errorMessage = `Request failed with status ${response.status}`;

      try {
        const errorData = await response.json();

        errorMessage = errorData.detail || errorData.message || errorMessage;
      } catch {
        const text = await response.text();

        if (text) {
          errorMessage = text;
        }
      }

      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    if (error.name === "AbortError") {
      throw error;
    }

    if (error instanceof TypeError) {
      throw new Error(
        "Could not connect to the backend. Make sure FastAPI is running.",
      );
    }

    throw error;
  }
}

async function apiPost(path, data) {
  try {
    const response = await fetch(API_BASE + path, {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(data),
    });

    if (!response.ok) {
      let errorMessage = `Request failed with status ${response.status}`;

      try {
        const errorData = await response.json();

        errorMessage = errorData.detail || errorData.message || errorMessage;
      } catch {
        const text = await response.text();

        if (text) {
          errorMessage = text;
        }
      }

      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error("Could not connect to the backend.");
    }

    throw error;
  }
}

/* =========================================================
   API HEALTH CHECK
========================================================= */

async function checkApiHealth() {
  try {
    const data = await apiGet("/health");

    if (data.status === "ok") {
      if (statusIndicator) {
        statusIndicator.classList.remove("offline");

        statusIndicator.classList.add("online");
      }

      if (apiStatusText) {
        if (data.mistral_configured) {
          apiStatusText.textContent = "AI System Online";
        } else {
          apiStatusText.textContent = "Movie System Online";
        }
      }
    }
  } catch (error) {
    console.error("API health check failed:", error);

    if (statusIndicator) {
      statusIndicator.classList.remove("online");

      statusIndicator.classList.add("offline");
    }

    if (apiStatusText) {
      apiStatusText.textContent = "Backend Offline";
    }
  }
}

/* =========================================================
   LOADING
========================================================= */

function showLoading(message = "Loading...") {
  if (loadingText) {
    loadingText.textContent = message;
  }

  if (loadingOverlay) {
    loadingOverlay.classList.remove("hidden");
  }
}

function hideLoading() {
  if (loadingOverlay) {
    loadingOverlay.classList.add("hidden");
  }
}

/* =========================================================
   TOAST
========================================================= */

function showToast(message) {
  if (!toast || !toastMessage) {
    console.log(message);
    return;
  }

  toastMessage.textContent = message;

  toast.classList.add("show");

  clearTimeout(toastTimeout);

  toastTimeout = setTimeout(() => {
    toast.classList.remove("show");
  }, 3500);
}

/* =========================================================
   VIEW NAVIGATION
========================================================= */

function showView(viewName) {
  const currentActive = document.querySelector(".view.active-view");

  if (currentActive && currentActive.id !== `${viewName}View`) {
    previousView = currentActive.id.replace("View", "");
  }

  views.forEach((view) => {
    view.classList.remove("active-view");
  });

  const target = document.getElementById(`${viewName}View`);

  if (target) {
    target.classList.add("active-view");
  }

  navItems.forEach((item) => {
    item.classList.remove("active");

    if (item.dataset.view === viewName) {
      item.classList.add("active");
    }
  });

  window.scrollTo({
    top: 0,
    behavior: "smooth",
  });

  closeMobileMenu();
}

/* =========================================================
   MOBILE MENU
========================================================= */

function openMobileMenu() {
  if (sidebar) {
    sidebar.classList.add("open");
  }

  if (sidebarOverlay) {
    sidebarOverlay.classList.add("show");
  }
}

function closeMobileMenu() {
  if (sidebar) {
    sidebar.classList.remove("open");
  }

  if (sidebarOverlay) {
    sidebarOverlay.classList.remove("show");
  }
}

/* =========================================================
   ESCAPE HTML
========================================================= */

function escapeHTML(value) {
  if (value === null || value === undefined) {
    return "";
  }

  const div = document.createElement("div");

  div.textContent = String(value);

  return div.innerHTML;
}

/* =========================================================
   FORMAT HELPERS
========================================================= */

function getYear(date) {
  if (!date) {
    return "";
  }

  return String(date).substring(0, 4);
}

function formatRuntime(minutes) {
  if (!minutes || Number(minutes) <= 0) {
    return "";
  }

  const hours = Math.floor(minutes / 60);

  const remainingMinutes = minutes % 60;

  if (hours > 0) {
    return remainingMinutes > 0
      ? `${hours}h ${remainingMinutes}m`
      : `${hours}h`;
  }

  return `${remainingMinutes}m`;
}

function getPosterUrl(movie) {
  if (movie.poster_url) {
    return movie.poster_url;
  }

  if (movie.poster_path) {
    return `${TMDB_IMAGE_BASE}${movie.poster_path}`;
  }

  return null;
}

/* =========================================================
   MOVIE CARD
========================================================= */

function createMovieCard(movie) {
  const card = document.createElement("div");

  card.className = "movie-card";

  const title = movie.title || "Untitled";

  const poster = getPosterUrl(movie);

  const year = getYear(movie.release_date);

  const rating = movie.vote_average;

  card.innerHTML = `

    <div class="poster-container">

      ${
        poster
          ? `
            <img
              src="${poster}"
              alt="${escapeHTML(title)}"
              loading="lazy"
            >
          `
          : `
            <div class="poster-placeholder">
              🎬
            </div>
          `
      }

      ${
        rating !== null && rating !== undefined && Number(rating) > 0
          ? `
            <div class="rating-badge">
              ⭐ ${Number(rating).toFixed(1)}
            </div>
          `
          : ""
      }

    </div>


    <div class="movie-info">

      <div
        class="movie-title"
        title="${escapeHTML(title)}"
      >
        ${escapeHTML(title)}
      </div>


      <div class="movie-meta">

        ${year || "Release date unavailable"}

      </div>

    </div>

  `;

  const image = card.querySelector("img");

  if (image) {
    image.addEventListener("error", () => {
      const container = image.parentElement;

      image.remove();

      if (container) {
        container.insertAdjacentHTML(
          "afterbegin",
          `
              <div class="poster-placeholder">
                🎬
              </div>
            `,
        );
      }
    });
  }

  card.addEventListener("click", async () => {
    if (movie.tmdb_id) {
      loadMovieDetails(movie.tmdb_id);

      return;
    }

    // TF-IDF recommendation may not
    // have a TMDB ID.
    // Search TMDB automatically.
    try {
      showLoading(`Finding ${title}...`);

      const data = await apiGet("/tmdb/search", {
        query: title,
      });

      const results = data.results || [];

      if (results.length > 0 && results[0].id) {
        await loadMovieDetails(results[0].id);
      } else {
        showToast("Movie details are not available.");
      }
    } catch (error) {
      console.error("Movie lookup error:", error);

      showToast("Could not load movie details.");
    } finally {
      hideLoading();
    }
  });

  return card;
}

/* =========================================================
   RENDER MOVIES
========================================================= */

function renderMovies(container, movies, emptyMessage = "Try another search.") {
  if (!container) {
    return;
  }

  container.innerHTML = "";

  if (!movies || movies.length === 0) {
    container.innerHTML = `

      <div class="empty-state">

        <div class="empty-icon">
          🎬
        </div>

        <h3>
          No Movies Found
        </h3>

        <p>
          ${escapeHTML(emptyMessage)}
        </p>

      </div>

    `;

    return;
  }

  movies.forEach((movie) => {
    const card = createMovieCard(movie);

    container.appendChild(card);
  });
}

/* =========================================================
   HOME MOVIES
========================================================= */

async function loadHomeMovies(category = "trending") {
  try {
    currentCategory = category;

    const title = category
      .replace(/_/g, " ")
      .replace(/\b\w/g, (character) => character.toUpperCase());

    const titleElement = document.getElementById("homeCategoryTitle");

    if (titleElement) {
      titleElement.textContent = `${title} Movies`;
    }

    categoryLinks.forEach((button) => {
      button.classList.remove("active-category");

      if (button.dataset.category === category) {
        button.classList.add("active-category");
      }
    });

    showLoading(`Loading ${title} Movies...`);

    const movies = await apiGet("/home", {
      category,
      limit: 24,
    });

    renderMovies(homeMoviesGrid, movies, "Please try refreshing.");
  } catch (error) {
    console.error("Home movies error:", error);

    renderMovies(homeMoviesGrid, [], error.message || "Could not load movies.");
  } finally {
    hideLoading();
  }
}

/* =========================================================
   SEARCH MOVIES
========================================================= */

async function searchMovies(query) {
  const cleanQuery = query.trim();

  if (cleanQuery.length < 2) {
    return;
  }

  if (searchController) {
    searchController.abort();
  }

  searchController = new AbortController();

  try {
    showLoading("Searching Movies...");

    const data = await apiGet(
      "/tmdb/search",
      {
        query: cleanQuery,
      },
      searchController.signal,
    );

    const results = data.results || [];

    const movies = results.map((movie) => ({
      tmdb_id: movie.id,
      title: movie.title || movie.name || "Untitled",

      poster_url: movie.poster_path
        ? `${TMDB_IMAGE_BASE}${movie.poster_path}`
        : null,

      release_date: movie.release_date,

      vote_average: movie.vote_average,
    }));

    renderMovies(searchMoviesGrid, movies, "Try another movie name.");

    if (resultsCount) {
      resultsCount.textContent =
        movies.length === 1 ? "1 movie found" : `${movies.length} movies found`;
    }

    renderSuggestions(movies.slice(0, 8));
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }

    console.error("Search error:", error);

    showToast(error.message || "Search failed. Please try again.");
  } finally {
    hideLoading();
  }
}

/* =========================================================
   SEARCH SUGGESTIONS
========================================================= */

function renderSuggestions(movies) {
  if (!suggestionsList || !suggestionsContainer) {
    return;
  }

  suggestionsList.innerHTML = "";

  if (!movies || movies.length === 0) {
    suggestionsContainer.classList.add("hidden");

    return;
  }

  movies.forEach((movie) => {
    const button = document.createElement("button");

    const year = getYear(movie.release_date);

    button.className = "suggestion-item";

    button.textContent = year ? `${movie.title} (${year})` : movie.title;

    button.addEventListener("click", () => {
      suggestionsContainer.classList.add("hidden");

      if (movieSearchInput) {
        movieSearchInput.value = movie.title;
      }

      loadMovieDetails(movie.tmdb_id);
    });

    suggestionsList.appendChild(button);
  });

  suggestionsContainer.classList.remove("hidden");
}

/* =========================================================
   MOVIE DETAILS
========================================================= */

async function loadMovieDetails(tmdbId) {
  try {
    showLoading("Loading Movie Details...");

    showView("details");

    if (movieDetailsContainer) {
      movieDetailsContainer.innerHTML = "";
    }

    if (tfidfRecommendations) {
      tfidfRecommendations.innerHTML = "";
    }

    if (genreRecommendations) {
      genreRecommendations.innerHTML = "";
    }

    const movie = await apiGet(`/movie/id/${tmdbId}`);

    renderMovieDetails(movie);

    // Recommendations load separately.
    // Even if they fail, movie details
    // remain visible.
    loadRecommendations(movie.title, tmdbId);
  } catch (error) {
    console.error("Movie details error:", error);

    showToast(error.message || "Could not load movie details.");
  } finally {
    hideLoading();
  }
}

function renderMovieDetails(movie) {
  if (!movieDetailsContainer) {
    return;
  }

  const genres = movie.genres || [];

  const genreHTML = genres
    .map(
      (genre) => `
          <span class="meta-pill">
            ${escapeHTML(genre.name)}
          </span>
        `,
    )
    .join("");

  const year = getYear(movie.release_date) || "Unknown";

  const runtime = formatRuntime(movie.runtime);

  const rating = movie.vote_average
    ? Number(movie.vote_average).toFixed(1)
    : null;

  const backdropStyle = movie.backdrop_url
    ? `
        style="
          background-image:
          url('${movie.backdrop_url}');
        "
      `
    : "";

  movieDetailsContainer.innerHTML = `

    <div class="movie-details-hero">

      <div
        class="details-backdrop"
        ${backdropStyle}
      ></div>

      <div
        class="details-overlay"
      ></div>

      <div class="movie-details-layout">


        <div>

          <div class="details-poster">

            ${
              movie.poster_url
                ? `
                  <img
                    src="${movie.poster_url}"
                    alt="${escapeHTML(movie.title)}"
                  >
                `
                : `
                  <div class="poster-placeholder">
                    🎬
                  </div>
                `
            }

          </div>

        </div>


        <div class="details-content">

          <p class="section-eyebrow">
            MOVIE DETAILS
          </p>

          <h1>
            ${escapeHTML(movie.title)}
          </h1>


          ${
            movie.tagline
              ? `
                <p class="movie-tagline">
                  "${escapeHTML(movie.tagline)}"
                </p>
              `
              : ""
          }


          <div class="details-meta">

            <span class="meta-pill">
              📅 ${year}
            </span>

            ${
              runtime
                ? `
                  <span class="meta-pill">
                    ⏱ ${runtime}
                  </span>
                `
                : ""
            }

            ${genreHTML}

          </div>


          <p class="overview">

            ${escapeHTML(movie.overview || "No overview available.")}

          </p>


          <div class="details-stats">

            ${
              rating
                ? `
                  <div class="detail-stat">

                    <span>
                      TMDB RATING
                    </span>

                    <strong>
                      ⭐ ${rating}/10
                    </strong>

                  </div>
                `
                : ""
            }


            ${
              movie.vote_count
                ? `
                  <div class="detail-stat">

                    <span>
                      VOTES
                    </span>

                    <strong>
                      ${Number(movie.vote_count).toLocaleString()}
                    </strong>

                  </div>
                `
                : ""
            }


            ${
              movie.status
                ? `
                  <div class="detail-stat">

                    <span>
                      STATUS
                    </span>

                    <strong>
                      ${escapeHTML(movie.status)}
                    </strong>

                  </div>
                `
                : ""
            }

          </div>


          <div class="details-actions">

            <button
              id="detailsAiButton"
              class="ai-analyze-small"
            >
              ✦ Analyze with AI
            </button>

          </div>

        </div>

      </div>

    </div>

  `;

  const aiButton = document.getElementById("detailsAiButton");

  if (aiButton) {
    aiButton.addEventListener("click", () => {
      showView("ai");

      if (aiMovieInput) {
        aiMovieInput.value = movie.title;

        aiMovieInput.focus();
      }
    });
  }
}

/* =========================================================
   RECOMMENDATIONS
========================================================= */

async function loadRecommendations(title, tmdbId) {
  /*
     Load TF-IDF recommendations
     and genre recommendations separately.

     This is important because one failure
     should not stop the other.
  */

  /* -------------------------
     TF-IDF RECOMMENDATIONS
  ------------------------- */

  try {
    const bundle = await apiGet("/movie/search", {
      query: title,
      tfidf_top_n: 12,
      genre_limit: 12,
    });

    const tfidfItems = bundle.tfidf_recommendations || [];

    /*
       Keep every TF-IDF result.

       Some local dataset titles may not
       have an exact TMDB match.
    */

    const tfidfMovies = tfidfItems.map((item) => {
      if (item.tmdb) {
        return item.tmdb;
      }

      return {
        title: item.title,
        poster_url: null,
        release_date: null,
        vote_average: null,
      };
    });

    renderMovies(
      tfidfRecommendations,
      tfidfMovies,
      "No similar movies were found in the local dataset.",
    );

    /* -------------------------
       GENRE RECOMMENDATIONS
    ------------------------- */

    const genreMovies = bundle.genre_recommendations || [];

    renderMovies(
      genreRecommendations,
      genreMovies,
      "No genre recommendations were found.",
    );
  } catch (error) {
    console.error("Recommendation bundle error:", error);

    /*
       TF-IDF recommendation bundle failed.
    */

    renderMovies(
      tfidfRecommendations,
      [],
      "Could not load similar movie recommendations.",
    );

    /*
       Try genre recommendations independently.
    */

    try {
      const genreMovies = await apiGet("/recommend/genre", {
        tmdb_id: tmdbId,
        limit: 12,
      });

      renderMovies(
        genreRecommendations,
        genreMovies,
        "No genre recommendations were found.",
      );
    } catch (genreError) {
      console.error("Genre recommendation error:", genreError);

      renderMovies(
        genreRecommendations,
        [],
        "Could not load genre recommendations.",
      );
    }
  }
}

/* =========================================================
   AI MOVIE ANALYZER
========================================================= */

async function analyzeMovie() {
  if (!aiMovieInput) {
    return;
  }

  const movieInput = aiMovieInput.value.trim();

  if (!movieInput) {
    showToast("Please enter a movie name or question.");

    return;
  }

  const originalButtonHTML = aiAnalyzeButton ? aiAnalyzeButton.innerHTML : "";

  try {
    showLoading("CineAI is analyzing the movie...");

    if (aiAnalyzeButton) {
      aiAnalyzeButton.disabled = true;

      aiAnalyzeButton.innerHTML = "Analyzing... ✦";
    }

    const result = await apiPost("/movie/extract", {
      movie_input: movieInput,
    });

    if (result.success && result.response) {
      if (aiResult) {
        aiResult.textContent = result.response;
      }

      if (aiResultContainer) {
        aiResultContainer.classList.remove("hidden");

        setTimeout(() => {
          aiResultContainer.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }, 200);
      }
    } else {
      showToast(result.error || "Analysis failed.");
    }
  } catch (error) {
    console.error("AI analysis error:", error);

    showToast(error.message || "AI analysis failed.");
  } finally {
    hideLoading();

    if (aiAnalyzeButton) {
      aiAnalyzeButton.disabled = false;

      aiAnalyzeButton.innerHTML = originalButtonHTML;
    }
  }
}

/* =========================================================
   COPY AI RESULT
========================================================= */

const copyAiResultButton = document.getElementById("copyAiResult");

if (copyAiResultButton) {
  copyAiResultButton.addEventListener("click", async () => {
    if (!aiResult || !aiResult.textContent.trim()) {
      showToast("There is no analysis to copy.");

      return;
    }

    try {
      await navigator.clipboard.writeText(aiResult.textContent);

      showToast("Analysis copied to clipboard!");
    } catch {
      showToast("Could not copy the analysis.");
    }
  });
}

/* =========================================================
   NAVIGATION EVENTS
========================================================= */

navItems.forEach((item) => {
  item.addEventListener("click", () => {
    showView(item.dataset.view);
  });
});

/* =========================================================
   CATEGORY EVENTS
========================================================= */

categoryLinks.forEach((button) => {
  button.addEventListener("click", () => {
    const category = button.dataset.category;

    showView("home");

    loadHomeMovies(category);
  });
});

/* =========================================================
   HERO BUTTONS
========================================================= */

const exploreButton = document.getElementById("exploreButton");

if (exploreButton) {
  exploreButton.addEventListener("click", () => {
    showView("search");

    if (movieSearchInput) {
      movieSearchInput.focus();
    }
  });
}

const analyzeButton = document.getElementById("analyzeButton");

if (analyzeButton) {
  analyzeButton.addEventListener("click", () => {
    showView("ai");

    if (aiMovieInput) {
      aiMovieInput.focus();
    }
  });
}

/* =========================================================
   REFRESH HOME
========================================================= */

const refreshHomeButton = document.getElementById("refreshHomeButton");

if (refreshHomeButton) {
  refreshHomeButton.addEventListener("click", () => {
    loadHomeMovies(currentCategory);
  });
}

/* =========================================================
   HOME SEARCH
========================================================= */

if (homeSearchButton) {
  homeSearchButton.addEventListener("click", () => {
    if (!homeSearchInput) {
      return;
    }

    const query = homeSearchInput.value.trim();

    if (!query) {
      showToast("Enter a movie name first.");

      return;
    }

    showView("search");

    if (movieSearchInput) {
      movieSearchInput.value = query;
    }

    searchMovies(query);
  });
}

if (homeSearchInput) {
  homeSearchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      if (homeSearchButton) {
        homeSearchButton.click();
      }
    }
  });
}

/* =========================================================
   MOVIE SEARCH
========================================================= */

if (movieSearchInput) {
  movieSearchInput.addEventListener("input", (event) => {
    clearTimeout(searchTimeout);

    const query = event.target.value.trim();

    if (query.length < 2) {
      if (suggestionsContainer) {
        suggestionsContainer.classList.add("hidden");
      }

      if (resultsCount) {
        resultsCount.textContent = "";
      }

      return;
    }

    searchTimeout = setTimeout(() => {
      searchMovies(query);
    }, 500);
  });

  movieSearchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      const query = movieSearchInput.value.trim();

      if (query.length >= 2) {
        clearTimeout(searchTimeout);

        searchMovies(query);
      }
    }
  });
}

/* =========================================================
   BACK BUTTON
========================================================= */

const backButton = document.getElementById("backButton");

if (backButton) {
  backButton.addEventListener("click", () => {
    showView(previousView === "details" ? "search" : previousView);
  });
}

/* =========================================================
   AI EVENTS
========================================================= */

if (aiAnalyzeButton) {
  aiAnalyzeButton.addEventListener("click", analyzeMovie);
}

if (aiMovieInput) {
  aiMovieInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.ctrlKey) {
      event.preventDefault();

      analyzeMovie();
    }
  });
}

/* =========================================================
   EXAMPLE BUTTONS
========================================================= */

document.querySelectorAll(".example-button").forEach((button) => {
  button.addEventListener("click", () => {
    if (aiMovieInput) {
      aiMovieInput.value = button.textContent.trim();
    }

    analyzeMovie();
  });
});

/* =========================================================
   MOBILE MENU EVENTS
========================================================= */

if (mobileMenuButton) {
  mobileMenuButton.addEventListener("click", () => {
    if (sidebar && sidebar.classList.contains("open")) {
      closeMobileMenu();
    } else {
      openMobileMenu();
    }
  });
}

if (sidebarOverlay) {
  sidebarOverlay.addEventListener("click", closeMobileMenu);
}

/* =========================================================
   INITIAL LOAD
========================================================= */

document.addEventListener("DOMContentLoaded", async () => {
  await checkApiHealth();

  loadHomeMovies("trending");
});
