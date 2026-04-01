# Modal Guide for AI Agents

## Core Modal Concepts for Agent Development

### 1. App Structure

Every Modal project starts with an `app`:

```python
import modal

app = modal.App("agent-project")

# Define container image
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("torch", "transformers", "vllm")
)
```

### 2. GPU Allocation

```python
@app.function(gpu="A100-80GB")
def run_inference():
    # GPU is available here
    pass

# Available GPUs: L4, A10G, A100, A100-80GB, H100
# Specify count: gpu="H100:2" for multi-GPU
```

### 3. Volumes (Persistent Storage)

Volumes persist data across function runs - essential for model caching:

```python
# Create once
volume = modal.Volume.from_name("agent-models", create_if_missing=True)

# Mount in functions
@app.function(volumes={"/models": volume})
def download_model():
    # Models cached at /models
    pass
```

### 4. Secrets (Environment Variables)

Store API keys securely:

```python
# Create via: modal secret create openai-secret OPENAI_API_KEY=xxx
secret = modal.Secret.from_name("openai-secret")

@app.function(secrets=[secret])
def call_llm():
    import os
    api_key = os.environ["OPENAI_API_KEY"]  # From secret
```

## Agent-Specific Patterns

### Pattern 1: Stateful Agent with Model Caching

```python
@app.cls(
    image=image,
    gpu="A100",
    volumes={"/models": model_volume},
    secrets=[api_secret],
    container_idle_timeout=300,  # Keep warm for 5 min
)
class Agent:
    @modal.enter()
    def load_model(self):
        """Called once when container starts. Model stays in memory."""
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            "/models/my-model",
            device_map="auto"
        )
        self.model.eval()

    @modal.method()
    def generate(self, prompt: str) -> str:
        """Each call reuses the cached model."""
        return self.model.generate(prompt)

    @modal.exit()
    def cleanup(self):
        """Called when container shuts down."""
        del self.model
```

### Pattern 2: Multi-Step Pipeline

```python
@app.function(
    image=image,
    gpu="A100",
    timeout=600,
)
def process_document(doc_path: str) -> dict:
    """Single atomic step in a pipeline."""
    # Extraction
    text = extract_text(doc_path)
    # Summarization
    summary = summarize(text)
    return {"text": text, "summary": summary}
```

### Pattern 3: Async Agent Loop

```python
@app.function(
    image=image,
    timeout=3600,  # 1 hour for long agent runs
    allow_concurrent_outputs=10,
)
async def run_agent_loop(initial_prompt: str, max_steps: int = 50):
    """Long-running agent with multiple LLM calls."""
    context = initial_prompt
    for step in range(max_steps):
        response = await call_llm(context)
        if is_complete(response):
            break
        context = update_context(context, response)
    return final_response
```

## Best Practices

### 1. Container Lifecycle

```python
# ✅ DO: Use @modal.enter() for expensive setup
# ✅ DO: Use @modal.exit() for cleanup
# ❌ DON'T: Load models inside the method (reloads every call)

@modal.enter()
def startup(self):
    self.model = load_model()  # One-time cost
```

### 2. Cold Start Optimization

```python
# For frequently-called agents, keep containers warm
@app.cls(
    container_idle_timeout=300,  # 5 min warm pool
    scaledown_window=60,        # Scale to 0 after 1 min idle
)
class WarmAgent:
    pass
```

### 3. Memory Management

```python
# For large models, use smaller sequences first
# then scale up once loaded
@app.function()
def first_run():
    # Small test to trigger model loading
    pass

@app.function()
def production_run():
    # Full workload
    pass
```

### 4. Error Handling & Retries

```python
from modal import retry

@app.function(
    retries=3,
    retry_delay_seconds=10,
)
@retry(max_retries=3)
def unreliable_step():
    # Auto-retries on failure
    pass
```

## Common Agent Architectures

### 1. Tool-Calling Agent

```python
@app.cls(image=image, gpu="H100")
class ToolAgent:
    def __init__(self):
        self.tools = [web_search, calculator, file_reader]

    def execute(self, task: str) -> str:
        while not self.is_complete(task):
            action = self.llm.plan(task, available_tools=self.tools)
            result = self.execute_tool(action)
            task = self.update(task, result)
        return task
```

### 2. RAG Agent

```python
@app.cls(image=image, gpu="A100")
class RAGAgent:
    @modal.enter()
    def load_index(self):
        self.vector_store = load_from_volume("/data/index")

    def query(self, question: str) -> str:
        docs = self.vector_store.similarity_search(question)
        context = format_docs(docs)
        return self.llm.answer(question, context)
```

### 3. Multi-Modal Agent

```python
@app.function(image=image, gpu="H100")
def analyze_image(image_path: str, query: str) -> str:
    model = load_multimodal_model()
    return model.analyze(image_path, query)
```

## Deployment

```bash
# Development (auto-reload)
modal serve agent_app.py

# Production deployment
modal deploy agent_app.py

# View logs
modal app logs agent-project

# Scale configuration
modal scaling set concurrency agent-app 100
```

## Debugging

```bash
# Interactive shell in container
modal shell agent_app.py::Agent.generate

# Local testing
modal run agent_app.py::Agent.generate --prompt "test"

# Volume inspection
modal volume ls
modal volume inspect agent-models
```

## Key Resources

- [Modal Docs](https://modal.com/docs)
- [GPU Guide](https://modal.com/docs/guide/gpu)
- [Volumes](https://modal.com/docs/guide/volumes)
- [Web Endpoints](https://modal.com/docs/guide/webhooks)
- [Modal Examples](https://github.com/modal-labs/modal-examples)
