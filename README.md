#  letta-mastermind-test

This project demonstrates an AI-powered **Mastermind** logic game using the **Letta framework** — a stateful agent system that connects to an **LLM (e.g., GPT-4o)** via the Letta API.
It runs multiple game sessions while tracking reasoning and memory through the Letta agent.

---

##  How to Run

### 1. Start the Letta Docker container

Run the following command to start the Letta service locally.
This container includes the Letta API server and its PostgreSQL database.

>  Replace the API key below with **your own** OpenAI API key before running.

```powershell
docker run -v E:\CAPSTONE\letta-ai\data\pgdata:/var/lib/postgresql/data `
  -p 8283:8283 `
  -e OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" `
  letta/letta:latest
```

**Explanation:**

* `-v E:\study\CAPSTONE\letta-ai\data\pgdata:/var/lib/postgresql/data` → Mounts your local PostgreSQL data folder for persistence
* `-p 8283:8283` → Exposes the Letta API server on port **8283**
* `-e OPENAI_API_KEY=...` → Passes your OpenAI key into the container environment
* `letta/letta:latest` → Uses the latest Letta image from Docker Hub

Once the container is running, the Letta API will be available at:
[http://localhost:8283](http://localhost:8283)

---

### 2. Clone the repository

```bash
git clone https://github.com/Dante1028/letta-mastermind-test.git
cd letta-mastermind-test/mastermind
```

---

### 3. Set up a Python environment

It’s recommended to use a virtual environment:

```bash
python -m venv venv
# Activate it:
.\venv\Scripts\activate        # Windows
# or
source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

---

### 4. Run the game

Execute the following command to start the full game simulation:

```bash
python run_full_game.py --model_type letta_auto --model gpt-4o --code_length 5 --num_colors 7 --num_runs 10 --letta_url http://localhost:8283
```

#### Argument details:

| Argument        | Description                                        | Example                 |
| --------------- | -------------------------------------------------- | ----------------------- |
| `--model_type`  | Type of model adapter (here: Letta automatic mode) | `letta_auto`            |
| `--model`       | OpenAI model used through Letta                    | `gpt-4o`                |
| `--code_length` | Number of positions in the secret code             | `5`                     |
| `--num_colors`  | Number of possible colors                          | `7`                     |
| `--num_runs`    | Number of game runs                                | `10`                    |
| `--letta_url`   | Base URL for your running Letta server             | `http://localhost:8283` |

---

## Project Structure

```
letta-mastermind-test/
├── mastermind/
│   ├── run_full_game.py        # Main entry point
│   ├── mastermind_game.py      # Core game logic
│   ├── memory_manager.py       # Memory handling via Letta
│   ├── config.py               # Configuration constants
│   └── requirements.txt
└── README.md
```

---

## Requirements

* **Python 3.9+**
* **Docker**
* **OpenAI API key** (passed to Letta container)
* **Letta** (via Docker, on port 8283)

---

## Example Output

When you run the game, you’ll see logs showing:

* The Letta agent making guesses
* Feedback on each guess (right color / right position)
* Memory/strategy updates between runs
* Final statistics for all runs

--
