import os
import pickle
import asyncio
import difflib

from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(
    os.path.join(BASE_DIR, ".env")
)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


# IMPORTANT:
# These must be normal strings.
# Do NOT use Markdown URLs like:
# [https://example.com](https://example.com)

TMDB_BASE = "https://api.themoviedb.org/3"

TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

TMDB_IMG_ORIGINAL = "https://image.tmdb.org/t/p/original"

MISTRAL_CHAT_URL = (
    "https://api.mistral.ai/v1/chat/completions"
)

MISTRAL_MODEL = os.getenv(
    "MISTRAL_MODEL",
    "mistral-small-latest"
)


if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY is missing.\n"
        "Add this to backend/.env:\n\n"
        "TMDB_API_KEY=your_tmdb_api_key"
    )


# =========================================================
# FILE PATHS
# =========================================================

DF_PATH = os.path.join(
    BASE_DIR,
    "df.pkl"
)

INDICES_PATH = os.path.join(
    BASE_DIR,
    "indices.pkl"
)

TFIDF_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "tfidf_matrix.pkl"
)

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "tfidf.pkl"
)


# =========================================================
# GLOBAL RESOURCES
# =========================================================

df: Optional[pd.DataFrame] = None

indices_obj: Any = None

tfidf_matrix: Any = None

tfidf_obj: Any = None

TITLE_TO_IDX: Optional[
    Dict[str, int]
] = None


# Reusable HTTP client

http_client: Optional[
    httpx.AsyncClient
] = None


# =========================================================
# PYDANTIC MODELS
# =========================================================

class TMDBMovieCard(BaseModel):

    tmdb_id: int

    title: str

    poster_url: Optional[str] = None

    release_date: Optional[str] = None

    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):

    tmdb_id: int

    title: str

    overview: Optional[str] = None

    tagline: Optional[str] = None

    release_date: Optional[str] = None

    runtime: Optional[int] = None

    vote_average: Optional[float] = None

    vote_count: Optional[int] = None

    popularity: Optional[float] = None

    status: Optional[str] = None

    original_language: Optional[str] = None

    poster_url: Optional[str] = None

    backdrop_url: Optional[str] = None

    genres: List[dict] = Field(
        default_factory=list
    )


class TFIDFRecItem(BaseModel):

    title: str

    score: float

    tmdb: Optional[
        TMDBMovieCard
    ] = None


class SearchBundleResponse(BaseModel):

    query: str

    movie_details: TMDBMovieDetails

    tfidf_recommendations: List[
        TFIDFRecItem
    ]

    genre_recommendations: List[
        TMDBMovieCard
    ]


class MovieExtractRequest(BaseModel):

    movie_input: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Movie name or movie-related description"
        ),
    )


class MovieExtractResponse(BaseModel):

    success: bool

    response: Optional[str] = None

    error: Optional[str] = None


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def normalize_title(
    title: str,
) -> str:

    title = str(title).strip().lower()

    replacements = {
        ":": " ",
        "-": " ",
        "_": " ",
        ".": " ",
        ",": " ",
        "'": "",
        '"': "",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "!": " ",
        "?": " ",
        "/": " ",
        "\\": " ",
    }

    for old, new in replacements.items():

        title = title.replace(
            old,
            new,
        )

    return " ".join(
        title.split()
    )


def make_img_url(
    path: Optional[str],
    size: str = "w500",
) -> Optional[str]:

    if not path:
        return None

    if size == "original":

        return (
            f"{TMDB_IMG_ORIGINAL}"
            f"{path}"
        )

    return (
        f"https://image.tmdb.org/t/p/"
        f"{size}{path}"
    )


def format_runtime(
    runtime: Optional[int],
) -> Optional[str]:

    if not runtime:
        return None

    hours = runtime // 60

    minutes = runtime % 60

    if hours > 0 and minutes > 0:

        return (
            f"{hours}h "
            f"{minutes}m"
        )

    if hours > 0:

        return f"{hours}h"

    return f"{minutes}m"


# =========================================================
# TMDB FUNCTIONS
# =========================================================

async def tmdb_get(
    path: str,
    params: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:

    global http_client

    if http_client is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "HTTP client is not initialized."
            ),
        )

    query_params = dict(
        params or {}
    )

    query_params["api_key"] = (
        TMDB_API_KEY
    )

    url = f"{TMDB_BASE}{path}"

    try:

        response = await http_client.get(
            url,
            params=query_params,
        )

    except httpx.TimeoutException as error:

        raise HTTPException(
            status_code=504,
            detail=(
                "TMDB request timed out. "
                "Please try again."
            ),
        ) from error

    except httpx.RequestError as error:

        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to connect to TMDB. "
                f"Network error: {str(error)}"
            ),
        ) from error

    if response.status_code == 401:

        raise HTTPException(
            status_code=401,
            detail=(
                "TMDB API key is invalid."
            ),
        )

    if response.status_code == 404:

        raise HTTPException(
            status_code=404,
            detail=(
                "Requested movie was not found "
                "on TMDB."
            ),
        )

    if response.status_code == 429:

        raise HTTPException(
            status_code=429,
            detail=(
                "TMDB rate limit reached. "
                "Please try again later."
            ),
        )

    if response.status_code >= 400:

        raise HTTPException(
            status_code=response.status_code,
            detail=(
                "TMDB API error: "
                f"{response.text}"
            ),
        )

    try:

        return response.json()

    except Exception as error:

        raise HTTPException(
            status_code=502,
            detail=(
                "TMDB returned an invalid response."
            ),
        ) from error


async def tmdb_cards_from_results(
    results: List[dict],
    limit: int = 20,
) -> List[TMDBMovieCard]:

    cards: List[
        TMDBMovieCard
    ] = []

    for movie in (
        results or []
    )[:limit]:

        movie_id = movie.get(
            "id"
        )

        if not movie_id:
            continue

        cards.append(

            TMDBMovieCard(

                tmdb_id=int(
                    movie_id
                ),

                title=(
                    movie.get("title")
                    or movie.get("name")
                    or "Untitled"
                ),

                poster_url=make_img_url(
                    movie.get(
                        "poster_path"
                    ),
                    "w500",
                ),

                release_date=(
                    movie.get(
                        "release_date"
                    )
                    or movie.get(
                        "first_air_date"
                    )
                ),

                vote_average=movie.get(
                    "vote_average"
                ),
            )
        )

    return cards


async def tmdb_movie_details(
    movie_id: int,
) -> TMDBMovieDetails:

    data = await tmdb_get(

        f"/movie/{movie_id}",

        {
            "language": "en-US",
        },
    )

    return TMDBMovieDetails(

        tmdb_id=int(
            data["id"]
        ),

        title=(
            data.get("title")
            or "Untitled"
        ),

        overview=data.get(
            "overview"
        ),

        tagline=data.get(
            "tagline"
        ),

        release_date=data.get(
            "release_date"
        ),

        runtime=data.get(
            "runtime"
        ),

        vote_average=data.get(
            "vote_average"
        ),

        vote_count=data.get(
            "vote_count"
        ),

        popularity=data.get(
            "popularity"
        ),

        status=data.get(
            "status"
        ),

        original_language=data.get(
            "original_language"
        ),

        poster_url=make_img_url(
            data.get(
                "poster_path"
            ),
            "w500",
        ),

        backdrop_url=make_img_url(
            data.get(
                "backdrop_path"
            ),
            "original",
        ),

        genres=(
            data.get(
                "genres",
                [],
            )
            or []
        ),
    )


async def tmdb_search_movies(
    query: str,
    page: int = 1,
) -> Dict[str, Any]:

    return await tmdb_get(

        "/search/movie",

        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(
    query: str,
) -> Optional[dict]:

    data = await tmdb_search_movies(
        query=query,
        page=1,
    )

    results = data.get(
        "results",
        [],
    )

    if not results:
        return None

    return results[0]


# =========================================================
# TF-IDF HELPERS
# =========================================================

def build_title_to_idx_map(
    indices: Any,
) -> Dict[str, int]:

    title_to_idx: Dict[
        str,
        int
    ] = {}

    try:

        for title, index in indices.items():

            normalized_title = (
                normalize_title(
                    title
                )
            )

            if (
                normalized_title
                not in title_to_idx
            ):

                title_to_idx[
                    normalized_title
                ] = int(index)

        return title_to_idx

    except Exception as error:

        raise RuntimeError(
            "indices.pkl must support "
            ".items()."
        ) from error


def get_local_idx_by_title(
    title: str,
) -> int:

    global TITLE_TO_IDX

    if TITLE_TO_IDX is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "TF-IDF title map "
                "is not initialized."
            ),
        )

    normalized_title = (
        normalize_title(title)
    )

    if normalized_title in TITLE_TO_IDX:

        return int(
            TITLE_TO_IDX[
                normalized_title
            ]
        )

    raise HTTPException(
        status_code=404,
        detail=(
            f"Movie '{title}' was not "
            "found in the local dataset."
        ),
    )


def find_best_local_title(
    title: str,
) -> Optional[str]:

    global df
    global TITLE_TO_IDX

    if (
        df is None
        or TITLE_TO_IDX is None
    ):

        return None

    normalized_query = (
        normalize_title(title)
    )

    # -----------------------------------------------------
    # 1. EXACT MATCH
    # -----------------------------------------------------

    if normalized_query in TITLE_TO_IDX:

        index = TITLE_TO_IDX[
            normalized_query
        ]

        try:

            return str(
                df.iloc[
                    int(index)
                ]["title"]
            )

        except Exception:

            return None

    # -----------------------------------------------------
    # 2. PARTIAL MATCH
    # -----------------------------------------------------

    partial_matches = []

    for (
        local_title,
        index,
    ) in TITLE_TO_IDX.items():

        if (
            normalized_query in local_title
            or local_title in normalized_query
        ):

            partial_matches.append(
                (
                    local_title,
                    index,
                )
            )

    if partial_matches:

        partial_matches.sort(

            key=lambda item: abs(
                len(item[0])
                - len(normalized_query)
            )
        )

        _, best_index = (
            partial_matches[0]
        )

        try:

            return str(
                df.iloc[
                    int(best_index)
                ]["title"]
            )

        except Exception:

            pass

    # -----------------------------------------------------
    # 3. FUZZY MATCH
    # -----------------------------------------------------

    close_matches = (
        difflib.get_close_matches(

            normalized_query,

            TITLE_TO_IDX.keys(),

            n=1,

            cutoff=0.70,
        )
    )

    if close_matches:

        best_match = (
            close_matches[0]
        )

        index = TITLE_TO_IDX[
            best_match
        ]

        try:

            return str(
                df.iloc[
                    int(index)
                ]["title"]
            )

        except Exception:

            return None

    return None


def tfidf_recommend_titles(
    query_title: str,
    top_n: int = 10,
) -> List[
    Tuple[str, float]
]:

    global df
    global tfidf_matrix

    if (
        df is None
        or tfidf_matrix is None
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "TF-IDF resources "
                "are not loaded."
            ),
        )

    local_title = (
        find_best_local_title(
            query_title
        )
    )

    if not local_title:

        return []

    movie_index = (
        get_local_idx_by_title(
            local_title
        )
    )

    try:

        query_vector = (
            tfidf_matrix[
                movie_index
            ]
        )

        scores = (
            tfidf_matrix
            @ query_vector.T
        ).toarray().ravel()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to calculate "
                f"TF-IDF similarity: {str(error)}"
            ),
        ) from error

    sorted_indices = (
        np.argsort(scores)[::-1]
    )

    recommendations: List[
        Tuple[str, float]
    ] = []

    seen_titles = set()

    for candidate_index in sorted_indices:

        candidate_index = int(
            candidate_index
        )

        if candidate_index == movie_index:
            continue

        try:

            movie_title = str(
                df.iloc[
                    candidate_index
                ]["title"]
            )

        except Exception:

            continue

        normalized_movie_title = (
            normalize_title(
                movie_title
            )
        )

        if (
            normalized_movie_title
            in seen_titles
        ):

            continue

        seen_titles.add(
            normalized_movie_title
        )

        score = float(
            scores[
                candidate_index
            ]
        )

        recommendations.append(
            (
                movie_title,
                score,
            )
        )

        if len(recommendations) >= top_n:
            break

    return recommendations


# =========================================================
# TMDB CARD ATTACHMENT
# =========================================================

async def attach_tmdb_card_by_title(
    title: str,
) -> Optional[TMDBMovieCard]:

    try:

        movie = await tmdb_search_first(
            title
        )

        if not movie:
            return None

        return TMDBMovieCard(

            tmdb_id=int(
                movie["id"]
            ),

            title=(
                movie.get("title")
                or title
            ),

            poster_url=make_img_url(
                movie.get(
                    "poster_path"
                )
            ),

            release_date=movie.get(
                "release_date"
            ),

            vote_average=movie.get(
                "vote_average"
            ),
        )

    except Exception as error:

        print(
            f"[WARNING] TMDB card failed "
            f"for '{title}': {error}"
        )

        return None


async def attach_tmdb_cards_parallel(
    recommendations: List[
        Tuple[str, float]
    ],
) -> List[
    TFIDFRecItem
]:

    if not recommendations:
        return []

    tasks = [

        attach_tmdb_card_by_title(
            title
        )

        for title, _ in recommendations
    ]

    cards = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    items: List[
        TFIDFRecItem
    ] = []

    for (
        (title, score),
        card,
    ) in zip(
        recommendations,
        cards,
    ):

        if isinstance(
            card,
            Exception,
        ):

            card = None

        items.append(

            TFIDFRecItem(

                title=title,

                score=score,

                tmdb=card,
            )
        )

    return items


# =========================================================
# GENRE RECOMMENDATIONS HELPER
# =========================================================

async def get_genre_recommendations(
    tmdb_id: int,
    genres: List[dict],
    limit: int,
) -> List[TMDBMovieCard]:

    if not genres:
        return []

    genre_id = genres[0].get(
        "id"
    )

    if not genre_id:
        return []

    try:

        discover = await tmdb_get(

            "/discover/movie",

            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )

        cards = (
            await tmdb_cards_from_results(

                discover.get(
                    "results",
                    [],
                ),

                limit=limit + 15,
            )
        )

        filtered_cards = [

            card

            for card in cards

            if card.tmdb_id != tmdb_id
        ]

        return filtered_cards[:limit]

    except Exception as error:

        print(
            "[WARNING] Genre recommendations "
            f"failed: {error}"
        )

        # IMPORTANT:
        # Return empty list instead of
        # crashing the entire endpoint.

        return []


# =========================================================
# MISTRAL AI
# =========================================================

async def analyze_movie_with_mistral(
    movie_input: str,
) -> str:

    global http_client

    if not MISTRAL_API_KEY:

        raise HTTPException(
            status_code=500,
            detail=(
                "MISTRAL_API_KEY is missing."
            ),
        )

    if http_client is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "HTTP client is not initialized."
            ),
        )

    system_prompt = """
You are CineAI, an intelligent movie analysis assistant.

Analyze movies accurately and clearly.

The user may provide:
- A movie title
- A movie title with questions
- A description of a movie

Create a well-structured movie report.

Use this structure when possible:

🎬 MOVIE
Title:
Release Year:
Genre:
Director:

⭐ RATINGS
Provide ratings only when reasonably certain.

📖 STORY OVERVIEW
Give a concise spoiler-free overview unless spoilers are requested.

🧠 THEMES & MEANING
Explain important themes and ideas.

🎭 CHARACTERS
Describe important characters.

🎥 WHAT MAKES IT SPECIAL
Discuss storytelling, direction, cinematography,
music, performances, or other strengths.

👍 WHO SHOULD WATCH IT
Describe the ideal audience.

✨ CINEAI VERDICT
Give a short final verdict.

Rules:
- Be factual.
- Do not invent information.
- Mention ambiguity if the movie title is ambiguous.
- Keep the answer engaging and readable.
- Use headings and bullet points.

Output Formatting Rules:


- Return clean plain text.
- Do NOT use Markdown.
- use bullet points.
- Do NOT use tables.
- Put each piece of information on a new line.


Use exactly this format:


Movie Title: <value>
Original Title: <value>
Release Year: <value>
Release Date: <value>
Genre: <value>
Director: <value>
Writers: <value>
Producers: <value>
Main Cast: <value>
Runtime: <value>
Language: <value>
Country of Origin: <value>
Production Companies: <value>
Budget: <value>
Box Office / Revenue: <value>
Ratings: <value>
Awards and Nominations: <value>
Plot Summary: <value>
Main Characters: <value>
Themes: <value>
Keywords: <value>
Movie Type: <value>
Short Summary: <value>

"""

    payload = {

        "model": MISTRAL_MODEL,

        "messages": [

            {
                "role": "system",
                "content": system_prompt,
            },

            {
                "role": "user",
                "content": movie_input,
            },
        ],

        "temperature": 0.7,

        "max_tokens": 1500,
    }

    headers = {

        "Authorization": (
            f"Bearer {MISTRAL_API_KEY}"
        ),

        "Content-Type": (
            "application/json"
        ),
    }

    try:

        response = await http_client.post(

            MISTRAL_CHAT_URL,

            json=payload,

            headers=headers,
        )

    except httpx.TimeoutException as error:

        raise HTTPException(
            status_code=504,
            detail=(
                "Mistral request timed out."
            ),
        ) from error

    except httpx.RequestError as error:

        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to connect to "
                f"Mistral AI: {str(error)}"
            ),
        ) from error

    if response.status_code >= 400:

        try:

            error_data = response.json()

        except Exception:

            error_data = response.text

        raise HTTPException(

            status_code=response.status_code,

            detail=(
                f"Mistral API error: "
                f"{error_data}"
            ),
        )

    try:

        data = response.json()

        content = (
            data["choices"][0]
            ["message"]["content"]
        )

    except Exception as error:

        raise HTTPException(
            status_code=502,
            detail=(
                "Invalid response received "
                "from Mistral AI."
            ),
        ) from error

    if not content:

        raise HTTPException(
            status_code=502,
            detail=(
                "Mistral returned an "
                "empty response."
            ),
        )

    return content


# =========================================================
# APPLICATION LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    global df
    global indices_obj
    global tfidf_matrix
    global tfidf_obj
    global TITLE_TO_IDX
    global http_client

    try:

        # -------------------------------------------------
        # CREATE HTTP CLIENT
        # -------------------------------------------------

        http_client = httpx.AsyncClient(

            timeout=httpx.Timeout(

                connect=10.0,

                read=30.0,

                write=30.0,

                pool=10.0,
            ),

            limits=httpx.Limits(

                max_connections=50,

                max_keepalive_connections=20,
            ),
        )

        # -------------------------------------------------
        # LOAD DATAFRAME
        # -------------------------------------------------

        with open(
            DF_PATH,
            "rb",
        ) as file:

            df = pickle.load(
                file
            )

        # -------------------------------------------------
        # LOAD INDICES
        # -------------------------------------------------

        with open(
            INDICES_PATH,
            "rb",
        ) as file:

            indices_obj = pickle.load(
                file
            )

        # -------------------------------------------------
        # LOAD TF-IDF MATRIX
        # -------------------------------------------------

        with open(
            TFIDF_MATRIX_PATH,
            "rb",
        ) as file:

            tfidf_matrix = pickle.load(
                file
            )

        # -------------------------------------------------
        # LOAD TF-IDF VECTORIZER
        # -------------------------------------------------

        with open(
            TFIDF_PATH,
            "rb",
        ) as file:

            tfidf_obj = pickle.load(
                file
            )

        # -------------------------------------------------
        # VALIDATE DATAFRAME
        # -------------------------------------------------

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            raise RuntimeError(
                "df.pkl does not contain "
                "a pandas DataFrame."
            )

        if "title" not in df.columns:

            raise RuntimeError(
                "df.pkl must contain "
                "a 'title' column."
            )

        # -------------------------------------------------
        # BUILD TITLE MAP
        # -------------------------------------------------

        TITLE_TO_IDX = (
            build_title_to_idx_map(
                indices_obj
            )
        )

        print()
        print("=" * 60)
        print("✓ Movie dataset loaded successfully")
        print(
            f"✓ Movies available: {len(df)}"
        )
        print(
            f"✓ Searchable titles: "
            f"{len(TITLE_TO_IDX)}"
        )
        print(
            "✓ TF-IDF model loaded successfully"
        )
        print(
            "✓ HTTP client initialized"
        )
        print(
            "✓ CineAI backend is ready"
        )
        print("=" * 60)
        print()

    except Exception as error:

        if http_client is not None:

            await http_client.aclose()

            http_client = None

        raise RuntimeError(
            "Failed to start CineAI backend: "
            f"{str(error)}"
        ) from error

    yield

    if http_client is not None:

        await http_client.aclose()

        http_client = None

    print(
        "CineAI backend shutting down..."
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(

    title="CineAI API",

    description=(
        "Movie discovery, recommendations, "
        "TMDB integration and Mistral AI "
        "movie analysis."
    ),

    version="5.0",

    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {

        "message": "Welcome to CineAI API",

        "status": "online",
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health():

    return {

        "status": "ok",

        "tmdb_configured": bool(
            TMDB_API_KEY
        ),

        "mistral_configured": bool(
            MISTRAL_API_KEY
        ),

        "dataset_loaded": (
            df is not None
        ),

        "tfidf_loaded": (
            tfidf_matrix is not None
        ),

        "http_client_ready": (
            http_client is not None
        ),

        "searchable_titles": (
            len(TITLE_TO_IDX)
            if TITLE_TO_IDX
            else 0
        ),
    }


# =========================================================
# HOME MOVIES
# =========================================================

@app.get(
    "/home",
    response_model=List[
        TMDBMovieCard
    ],
)
async def home(

    category: str = Query(
        "trending"
    ),

    limit: int = Query(
        24,
        ge=1,
        le=50,
    ),
):

    if category == "trending":

        data = await tmdb_get(

            "/trending/movie/day",

            {
                "language": "en-US",
            },
        )

    elif category in {

        "popular",
        "top_rated",
        "upcoming",
        "now_playing",

    }:

        data = await tmdb_get(

            f"/movie/{category}",

            {
                "language": "en-US",
                "page": 1,
            },
        )

    else:

        raise HTTPException(

            status_code=400,

            detail=(
                "Invalid category. Use: "
                "trending, popular, top_rated, "
                "now_playing or upcoming."
            ),
        )

    return await tmdb_cards_from_results(

        data.get(
            "results",
            [],
        ),

        limit=limit,
    )


# =========================================================
# TMDB SEARCH
# =========================================================

@app.get("/tmdb/search")
async def tmdb_search(

    query: str = Query(
        ...,
        min_length=1,
        max_length=200,
    ),

    page: int = Query(
        1,
        ge=1,
        le=10,
    ),
):

    return await tmdb_search_movies(

        query=query,

        page=page,
    )


# =========================================================
# MOVIE DETAILS
# =========================================================

@app.get(
    "/movie/id/{tmdb_id}",
    response_model=TMDBMovieDetails,
)
async def movie_details_route(

    tmdb_id: int,
):

    return await tmdb_movie_details(
        tmdb_id
    )


# =========================================================
# GENRE RECOMMENDATIONS
# =========================================================

@app.get(
    "/recommend/genre",
    response_model=List[
        TMDBMovieCard
    ],
)
async def recommend_genre(

    tmdb_id: int = Query(
        ...
    ),

    limit: int = Query(
        18,
        ge=1,
        le=50,
    ),
):

    try:

        details = await tmdb_movie_details(
            tmdb_id
        )

        return await get_genre_recommendations(

            tmdb_id=tmdb_id,

            genres=details.genres,

            limit=limit,
        )

    except HTTPException as error:

        # Don't return 502 for a temporary
        # recommendation failure.

        print(
            f"[WARNING] Genre route error: "
            f"{error.detail}"
        )

        return []

    except Exception as error:

        print(
            f"[WARNING] Genre route failed: "
            f"{error}"
        )

        return []


# =========================================================
# TF-IDF RECOMMENDATIONS
# =========================================================

@app.get("/recommend/tfidf")
async def recommend_tfidf(

    title: str = Query(
        ...,
        min_length=1,
    ),

    top_n: int = Query(
        10,
        ge=1,
        le=50,
    ),
):

    print()
    print("=" * 60)
    print(
        f"TF-IDF REQUEST: {title}"
    )

    best_match = (
        find_best_local_title(
            title
        )
    )

    print(
        f"BEST LOCAL MATCH: "
        f"{best_match}"
    )

    recommendations = (
        tfidf_recommend_titles(

            title,

            top_n=top_n,
        )
    )

    print(
        f"RECOMMENDATIONS FOUND: "
        f"{len(recommendations)}"
    )

    print("=" * 60)

    return [

        {
            "title": movie_title,
            "score": score,
        }

        for movie_title, score
        in recommendations
    ]


# =========================================================
# MOVIE SEARCH + RECOMMENDATION BUNDLE
# =========================================================

@app.get(
    "/movie/search",
    response_model=SearchBundleResponse,
)
async def search_bundle(

    query: str = Query(

        ...,

        min_length=1,

        max_length=200,
    ),

    tfidf_top_n: int = Query(

        12,

        ge=1,

        le=30,
    ),

    genre_limit: int = Query(

        12,

        ge=1,

        le=30,
    ),
):

    print()
    print("=" * 70)
    print(
        f"MOVIE SEARCH REQUEST: {query}"
    )

    # -----------------------------------------------------
    # SEARCH TMDB
    # -----------------------------------------------------

    best_match = (
        await tmdb_search_first(
            query
        )
    )

    if not best_match:

        raise HTTPException(

            status_code=404,

            detail=(
                f"No movie found for "
                f"'{query}'."
            ),
        )

    tmdb_id = int(
        best_match["id"]
    )

    # -----------------------------------------------------
    # GET MOVIE DETAILS
    # -----------------------------------------------------

    details = (
        await tmdb_movie_details(
            tmdb_id
        )
    )

    print(
        f"SELECTED MOVIE: "
        f"{details.title}"
    )

    # -----------------------------------------------------
    # TF-IDF RECOMMENDATIONS
    # -----------------------------------------------------

    local_recommendations: List[
        Tuple[str, float]
    ] = []

    try:

        print(
            f"Trying TF-IDF: "
            f"{details.title}"
        )

        local_recommendations = (
            tfidf_recommend_titles(

                details.title,

                top_n=tfidf_top_n,
            )
        )

        print(
            f"TF-IDF FOUND: "
            f"{len(local_recommendations)}"
        )

    except Exception as error:

        print(
            f"[WARNING] TF-IDF failed "
            f"for TMDB title: {error}"
        )

        try:

            print(
                f"Trying original query: "
                f"{query}"
            )

            local_recommendations = (
                tfidf_recommend_titles(

                    query,

                    top_n=tfidf_top_n,
                )
            )

        except Exception as fallback_error:

            print(
                f"[WARNING] TF-IDF fallback "
                f"failed: {fallback_error}"
            )

            local_recommendations = []

    # -----------------------------------------------------
    # RUN POSTER FETCHING AND GENRE RECOMMENDATIONS
    # IN PARALLEL
    # -----------------------------------------------------

    tfidf_task = (
        attach_tmdb_cards_parallel(
            local_recommendations
        )
    )

    genre_task = (
        get_genre_recommendations(

            tmdb_id=details.tmdb_id,

            genres=details.genres,

            limit=genre_limit,
        )
    )

    results = await asyncio.gather(

        tfidf_task,

        genre_task,

        return_exceptions=True,
    )

    # -----------------------------------------------------
    # SAFE TF-IDF RESULTS
    # -----------------------------------------------------

    if isinstance(
        results[0],
        Exception,
    ):

        print(
            f"[WARNING] Poster attachment "
            f"failed: {results[0]}"
        )

        tfidf_items = []

    else:

        tfidf_items = results[0]

    # -----------------------------------------------------
    # SAFE GENRE RESULTS
    # -----------------------------------------------------

    if isinstance(
        results[1],
        Exception,
    ):

        print(
            f"[WARNING] Genre recommendation "
            f"failed: {results[1]}"
        )

        genre_recommendations = []

    else:

        genre_recommendations = results[1]

    print(
        f"TF-IDF RESULTS: "
        f"{len(tfidf_items)}"
    )

    print(
        f"GENRE RESULTS: "
        f"{len(genre_recommendations)}"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # RETURN RESPONSE
    # -----------------------------------------------------

    return SearchBundleResponse(

        query=query,

        movie_details=details,

        tfidf_recommendations=(
            tfidf_items
        ),

        genre_recommendations=(
            genre_recommendations
        ),
    )


# =========================================================
# AI MOVIE ANALYZER
# =========================================================

@app.post(
    "/movie/extract",
    response_model=MovieExtractResponse,
)
async def movie_extract(

    request: MovieExtractRequest,
):

    try:

        analysis = (
            await analyze_movie_with_mistral(

                request.movie_input
                .strip()
            )
        )

        return MovieExtractResponse(

            success=True,

            response=analysis,
        )

    except HTTPException as error:

        return MovieExtractResponse(

            success=False,

            error=str(
                error.detail
            ),
        )

    except Exception as error:

        return MovieExtractResponse(

            success=False,

            error=(
                "AI analysis failed: "
                f"{str(error)}"
            ),
        )