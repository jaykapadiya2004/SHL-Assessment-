# SHL Assessment Recommender

A conversational FastAPI agent that recommends SHL Individual Test Solutions based on hiring context.

## Architecture

```
POST /chat  →  Claude (claude-sonnet-4-5)  →  JSON response
                    ↑
           System prompt with full
           SHL catalog embedded
```

**Design decisions:**
- **In-context retrieval**: All 52 catalog entries are embedded directly in the system prompt. At ~8K tokens this fits comfortably within Claude's context and avoids vector search latency — critical for the 30-second timeout constraint.
- **Stateless**: Full conversation history is passed each turn per the spec.
- **Schema enforcement**: Claude is prompted to return only JSON. A robust parser strips markdown fences and falls back gracefully on parse errors.
- **Hallucination guard**: All recommendations are validated against a set of known catalog URLs/names. Any invented URL is silently dropped.

## Setup

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Test
python test_api.py
```

### Deploy to Render (Free Tier)

1. Push this directory to a GitHub repo
2. Go to [render.com](https://render.com) → New Web Service → Connect repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `ANTHROPIC_API_KEY` = your key
6. Deploy — `/health` and `/chat` will be live

### Deploy to Railway

```bash
railway login
railway init
railway add --service
railway env set ANTHROPIC_API_KEY=sk-ant-...
railway up
```

### Deploy with Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... shl-recommender
```

## API

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Catalog

52 SHL Individual Test Solutions covering:
- **A** Ability & Aptitude (Verify series: Numerical, Verbal, Inductive, Deductive, G+)
- **B** Biodata & Situational Judgement
- **C** Competencies (UCF Questionnaire)
- **D** Development & 360
- **E** Assessment Exercises (Smart Interview)
- **K** Knowledge & Skills (Java, Python, SQL, JavaScript, C++, .NET, etc.)
- **P** Personality & Behavior (OPQ32, OPQ32r, MQ)
- **S** Simulations (Automata coding, Customer Contact, Call Center)

## Agent Behavior

| Situation | Agent Action |
|-----------|-------------|
| Vague query ("I need an assessment") | Asks clarifying question (role, level) |
| Enough context | Returns 1–10 ranked recommendations |
| User refines constraints | Updates shortlist without starting over |
| Comparison request | Compares using catalog data only |
| Off-topic / prompt injection | Politely refuses |
| Hallucinated URL | Silently dropped by validator |
