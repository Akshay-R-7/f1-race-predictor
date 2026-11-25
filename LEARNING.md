# Learning guide

This guide walks through the project in the order the data flows, with each idea explained in plain language before the code. Read one part, open the file it points to, then try the exercise at the end of that part. Budget about a week if you do every exercise.

**Before you start:** run the quick start in the README once, so every file below exists on your machine.

---

## Part 1: The big idea

We want a probability for each driver winning a race, like "Norris 50%, Piastri 24%".

There are two ways to get it:

1. **Learn it directly:** look at past races and learn "pole sitters win about 57% of the time".
2. **Simulate it:** predict how fast each car is, then play the race out many times with randomness (safety cars, retirements, pit stops) and count who wins.

The second way is called a **Monte Carlo simulation**. It's like rolling dice 100,000 times to learn how often you get a six, except each "roll" is a whole race. The project builds the second way and compares it honestly against the first.

---

## Part 2: Data, and why we validate it (`f1sim/data/`)

**Concept.** Real-world data is messy: a driver might appear with grid position 0, or a lap time of 40 seconds slower than everyone because of a timing glitch. If bad rows reach the model, the model learns nonsense quietly.

**How the code handles it**
- `schema.py` describes what a valid row looks like with **Pydantic** models: a grid slot from 1 to 30, a three-letter driver code, and "a driver who didn't finish can't have a finishing position". Pydantic checks every row against these rules.
- `db.py` → `write()` validates rows **before** inserting them. Good rows go into SQLite in **one transaction** (all or nothing). Bad rows are written to `rejected_rows.jsonl` with the reason, so nothing disappears silently.
- `jolpica_loader.py` gets real data for free. Three ideas worth reading:
  - **Pagination:** the API returns at most 100 records per page, so `get_all()` keeps asking for the next `offset` until it reaches `total`, then merges pages that belong to the same race.
  - **Caching and rate limits:** `Client.get()` saves every response to disk and waits between real requests. If the server says "too many requests" (HTTP 429), it waits and tries again with a longer pause each time. This is called **exponential backoff**.
  - **Cleaning without labels:** there's no safety-car flag, so `pace_deltas()` uses simple rules: drop laps where the whole field is slow, and drop one driver's laps that are far slower than their own normal pace.
- `fastf1_loader.py` is the richer, paid alternative. Look at `tyre_degradation()`: for each stint it fits a straight line of lap time against tyre age. The **slope** of that line is how many seconds per lap the tyres lose.
- `sample.py` makes realistic fake seasons, so everything works without internet access.

**Exercise:** add a rule to `Result` that a driver can't have more pit stops than laps completed. Add a test for it in `tests/test_data.py`.

---

## Part 3: Features, and the classic mistake (`f1sim/features.py`)

**Concept: data leakage.** If the model is allowed to see information from the race it's predicting, it looks brilliant in testing and fails in real life. The most common way this happens is a rolling average that includes the current race.

**How the code avoids it**
- `_past_mean()` uses `shift(1)`: it moves the data down one row first, so race N's average only uses races before N.
- `time_split()` trains on older seasons and tests on newer ones. Never shuffle randomly across seasons.
- `tests/test_features.py::test_features_do_not_use_the_current_race` proves it. It changes one race's results drastically and checks that race's features stay exactly the same.

**Exercise:** delete `.shift(1)` in `_past_mean`, run `pytest tests/test_features.py`, and watch the leakage test fail. Put it back.

---

## Part 4: The pace model (`f1sim/model.py`)

**Concept: one network, three answers.** A neural network can have a shared "body" that learns general things ("this car is fast at this kind of track") and several small "heads" that each answer one question:
- **pace:** how many seconds per lap faster or slower than the field (regression),
- **dnf:** the chance of retiring (a probability, so it ends in a sigmoid),
- **deg:** tyre wear (regression).

**Details worth knowing**
- `Normalization` rescales inputs so a grid slot (1–20) and a gap in seconds (0–3) are on similar scales.
- **Masking with sample weights:** a car that retired on lap 3 has no pace measurement. Its weight for the pace loss is set to 0, so the missing value doesn't teach the model anything.
- **Early stopping:** training stops when the validation loss stops improving, and the best weights are restored.
- **Calibration:** `fit_post_processing()` shifts the retirement probabilities so their average matches what actually happened in the validation season.

**Exercise:** remove the `Dropout` layer and compare the pace error on test. Did it change? Why might dropout matter less on a small model?

---

## Part 5: The C++ simulator (`cpp/`)

**Concept.** Python is slow at tight loops: 100,000 races × 20 drivers × 57 laps is 114 million small steps. C++ compiles to machine code, so the same loop is much faster. Threads let several CPU cores share the work.

Read `cpp/src/racesim.cpp` → `simulate_once()` top to bottom. One race works like this:
1. **Start:** each car starts behind the car ahead by `grid_gap` seconds, and gets today's form, which is its predicted pace plus some randomness.
2. **Each lap:** a lap time is pace, plus tyre wear × tyre age, plus noise. Each car may retire (`hazard`), may pit, and a safety car may bunch everyone up.
3. **Overtaking:** if a faster car catches the one ahead, it only gets past with probability `1 - overtake_difficulty`; otherwise it's stuck just behind. This is why pole matters more at Monaco.
4. **Pit timing:** a car pits after roughly `sqrt(2 × pit_loss / deg)` laps. That's where the time lost to old tyres equals the time a stop costs.

**Threads, in `simulate()`.** The races are split evenly between threads. Each thread has **its own random generator and its own counters**, so threads never write to the same memory and need no locks. Results are added up after `join()`. Same seed + same thread count = same answer, which the tests check.

**pybind11, in `bindings.cpp`.** This turns the C++ function into a Python function. `gil_scoped_release` lets Python carry on while C++ works.

**The speed fix.** The first version created new vectors and fully re-sorted the field on every lap. Reusing buffers and using insertion sort (fast when the order barely changes) made it 26% faster with identical results. Always measure before and after.

**Exercise:** add a "virtual safety car" (everyone loses 5 seconds but gaps stay the same) with its own probability per lap. Add a GoogleTest test that it changes nothing when its probability is 0.

---

## Part 6: Judging probabilities (`f1sim/metrics.py`)

Accuracy alone is misleading for probabilities. Say forecaster A gives the winner 51% and B gives them 99%. Both "picked the winner", but only B was confident, and only B is badly punished when it's wrong.

- **Log loss:** `-log(probability given to the actual winner)`. 50% → 0.69, 10% → 2.30, 1% → 4.6. Confident mistakes cost a lot.
- **Brier score:** the squared difference between your probabilities and what happened, added up over drivers. Gentler than log loss.
- **Calibration:** take everything you called "about 25%". Did about 25% of those actually win? `results/calibration.png` plots this; points near the diagonal mean calibrated.

**Why "pole always wins" has a log loss of 4.0 even though it picks the winner 57% of the time:** when the pole sitter doesn't win, it gave the real winner almost 0%, which log loss punishes enormously.

---

## Part 7: What broke, the most important part

Read "What broke and why" in the README. In short:
- The simulator underrated starting position (39% vs 57% for pole).
- Tuning on validation helped, but the tuner kept choosing the edge of the range. That means the simulator's model is missing something, not that you should widen the range.
- When asked to blend simulator and baseline, validation chose 0% simulator.

**The lesson:** a beautiful system that loses to a one-line baseline isn't a failure if you measure it honestly and can explain why. That's what research teams want to hear about.

**Exercise (the big one):** run the pipeline on real Jolpica data. Does the simulator beat the grid baseline on real races? Write down why, either way.

---

## Part 8: The dashboard (`api/`, `web/`)

- **FastAPI** (`api/main.py`): each `@app.get` function becomes a URL. `/api/simulate` validates its input with Pydantic, for example rejecting two drivers on the same grid slot with a 422 error.
- **React + TypeScript** (`web/src/App.tsx`): `useEffect` loads data when the page opens. `types.ts` describes the API's JSON, so TypeScript catches typos like `win_probb` before you run anything.
- **Tests:** `lib.test.ts` tests pure functions; `App.test.tsx` fakes `fetch` and checks the page shows the data, and an error message when the API is down.

**Exercise:** add a "what if" form that posts to `/api/simulate` and shows the result.

---

## Part 9: Docker and CI

- `Dockerfile.api` builds the C++ module inside the image, then copies only what's needed into a small final image (a **multi-stage build**).
- `docker-compose.yml` starts the API and the dashboard together; nginx forwards `/api` to the backend.
- `.github/workflows/ci.yml` runs three jobs on every push: the C++ tests, the Python tests, and the TypeScript type check plus Vitest.

---

## Part 10: Quantisation (`quantize.py`)

**Concept.** Neural networks normally store numbers as 32-bit floats. **int8 quantisation** stores them as 8-bit integers: about 4× less memory, often faster on phones and AI chips. A small **representative dataset** tells the converter which value ranges to expect.

**What happened here:** the file was only 35% smaller and slightly slower, and accuracy dropped. The model is tiny (16 KB), so fixed overhead dominates. Quantisation pays off on big models. Being able to say *when* a technique helps is worth more than saying you used it.

---

## Making it yours (do at least two before interviews)

1. Run it on real data and update the README results.
2. Add the virtual safety car (Part 5).
3. Add a learned "who wins" baseline: a softmax over drivers using qualifying gap and form.
4. Build the pit-strategy agent: wrap the simulator as a Gymnasium environment and train PPO.

## Questions to be ready for

- *Why split by season instead of randomly?* Random splits leak future information, for example a car's 2024 upgrade, into training.
- *Why C++?* 114 million lap steps per forecast. C++ was 16× faster on one thread and 28× on two.
- *How do threads avoid race conditions here?* Each thread has its own random generator and counters; results are combined after all threads finish.
- *Why did your model lose to the baseline, and what did you do?* See Part 7.
- *Is your model calibrated? How do you know?* Calibration error 0.008, plus the calibration plot.
