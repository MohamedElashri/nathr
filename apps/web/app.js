"use strict";

const el = (id) => document.getElementById(id);
const modeButtons = [...document.querySelectorAll("[data-mode]")];
const modes = {
  chat: {
    name: "Chat", subtitle: "Send messages to your local model",
    empty: "Start a conversation", detail: "Each reply uses the messages in this conversation.",
    prompt: "Write a message", run: "Send",
    starters: ["Why is the sky blue?", "Explain how a solar panel works.", "Write a short story about a train."],
  },
  completion: {
    name: "Completion", subtitle: "Continue a prompt with your local model",
    empty: "Give the model a starting point", detail: "The model generates the text that follows your prompt.",
    prompt: "Enter a prompt", run: "Generate",
    starters: ["Once upon a time", "The first step in making bread is"],
  },
  tensor: {
    name: "Named tensors", subtitle: "Inspect the outputs exposed by your checkpoint",
    empty: "Run a tensor request", detail: "Choose an output and enter text or token IDs.",
    prompt: "Enter the input text", run: "Run",
  },
};

let model, mode, active, tensor;
let history = [];
const prompt = el("prompt");
const results = el("results");
const form = el("composer");

function status(message, error = false) {
  el("status").textContent = message;
  el("status").classList.toggle("error", error);
}

function scrollToEnd() {
  results.scrollTop = results.scrollHeight;
}

function message(role, text) {
  const group = document.createElement("article");
  group.className = "message " + role;
  const label = document.createElement("h2");
  label.textContent = { user: "You", assistant: "Nathr", prompt: "Prompt" }[role];
  const body = document.createElement("p");
  body.textContent = text;
  group.append(label, body);
  el("transcript").append(group);
  return { group, body };
}

function inputControls() {
  const raw = el("input-kind").value === "tokens";
  const isTensor = mode === "tensor";
  el("input-tools").hidden = mode === "chat";
  el("input-kind-label").hidden = !model.text;
  el("output-label").hidden = !isTensor;
  el("pair-label").hidden = !isTensor || raw || !model.inputs.includes("token_type_ids");
  const segment = raw && model.inputs.includes("token_type_ids");
  const decoder = model.inputs.includes("decoder_tokens");
  el("ids-label").hidden = !isTensor || (!segment && !decoder);
  el("ids-title").textContent = segment ? "Segment IDs" : "Decoder prefix IDs";
  const label = mode !== "chat" && raw ? "Enter comma-separated token IDs" : modes[mode].prompt;
  prompt.placeholder = label;
  el("prompt-label").textContent = label;
  el("suggestions").hidden = mode !== "chat" && raw;
}

function showMode(next) {
  if (active) return;
  mode = next;
  const view = modes[mode];
  for (const button of modeButtons)
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  el("heading").textContent = view.name;
  el("subtitle").textContent = view.subtitle;
  el("empty-title").textContent = view.empty;
  el("empty-detail").textContent = view.detail;
  el("suggestions").replaceChildren(...(view.starters || []).map((text) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.addEventListener("click", () => {
      prompt.value = text;
      prompt.focus();
    });
    return button;
  }));
  el("transcript").replaceChildren();
  if (mode === "chat")
    for (const item of history) message(item.role, item.content);
  el("empty").hidden = mode === "chat" && history.length > 0;
  el("reset").hidden = mode !== "chat";
  el("generation-settings").hidden = mode === "tensor";
  el("tensor-result").hidden = true;
  el("run").textContent = view.run;
  el("pair").value = el("extra-ids").value = prompt.value = "";
  el("input-kind").value = model.text ? "text" : "tokens";
  inputControls();
  status("");
  scrollToEnd();
}

function parseIds(text, label, limit) {
  const fields = text.split(/[\s,]+/).filter(Boolean);
  if (!fields.length || fields.some((id) => !/^\d+$/.test(id) || Number(id) >= limit))
    throw new Error(label + " must contain valid nonnegative integers.");
  return fields.map(Number);
}

function payload(text) {
  const raw = mode !== "chat" && el("input-kind").value === "tokens";
  if (mode === "tensor") {
    const data = raw ? { tokens: parseIds(text, "Token IDs", model.vocabulary) } : { prompt: text };
    const pair = el("pair").value;
    const extra = el("extra-ids").value.trim();
    if (!el("pair-label").hidden && pair) data.text_pair = pair;
    if (!el("ids-label").hidden && extra) {
      if (raw && model.inputs.includes("token_type_ids"))
        data.token_type_ids = parseIds(extra, "Segment IDs", model.vocabulary);
      else
        data.decoder_tokens = parseIds(extra, "Decoder prefix IDs", model.vocabulary);
    }
    return { ...data, output: el("output").value };
  }
  for (const id of ["max-tokens", "temperature", "seed"]) {
    if (!el(id).reportValidity()) throw new Error("Check the generation settings.");
  }
  const temperature = Number(el("temperature").value);
  const data = { max_new_tokens: Number(el("max-tokens").value), greedy: temperature === 0, stream: true };
  if (temperature) data.temperature = temperature;
  if (el("seed").value) {
    const seed = Number(el("seed").value);
    if (!Number.isSafeInteger(seed)) throw new Error("Enter a seed that fits in a JavaScript integer.");
    data.seed = seed;
  }
  if (mode === "chat") data.messages = [...history, { role: "user", content: text }];
  else if (raw) {
    data.tokens = parseIds(text, "Token IDs", model.vocabulary);
    if (model.text) data.decode = true;
  } else data.prompt = text;
  return data;
}

async function streamEvents(body, receive) {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let pending = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += value;
      let end;
      while ((end = pending.indexOf("\n\n")) !== -1) {
        const frame = pending.slice(0, end);
        pending = pending.slice(end + 2);
        if (!frame.startsWith("data: ")) continue;
        const data = frame.slice(6);
        if (data === "[DONE]") return;
        receive(JSON.parse(data));
      }
    }
    throw new Error("The response ended before generation completed.");
  } finally {
    await reader.cancel().catch(() => {});
  }
}

function busy(value) {
  for (const control of [...form.elements, ...modeButtons, el("reset")]) control.disabled = value;
  el("run").hidden = value;
  el("stop").hidden = !value;
  el("stop").disabled = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (active) return;
  if (!prompt.value.trim()) {
    status("Enter text or token IDs to run inference.", true);
    return;
  }
  const input = prompt.value;
  let data;
  try {
    data = payload(input);
  } catch (error) {
    status(error.message, true);
    return;
  }
  const controller = new AbortController();
  active = controller;
  busy(true);
  status(mode === "tensor" ? "Running inference…" : "Generating…");
  const isTensor = mode === "tensor";
  let inputMessage, outputMessage, reply = "", tokens = [];
  if (!isTensor) {
    el("empty").hidden = true;
    if (mode === "completion") el("transcript").replaceChildren();
    inputMessage = message(mode === "chat" ? "user" : "prompt", input);
    outputMessage = message("assistant", "Waiting for the model…");
    scrollToEnd();
  }
  const recordReply = () => {
    if (mode === "chat") {
      history.push({ role: "user", content: input }, { role: "assistant", content: reply });
      prompt.value = "";
    }
  };
  try {
    const response = await fetch("/infer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data), signal: controller.signal,
    });
    if (!response.ok) {
      const fallback = "Inference failed (HTTP " + response.status + ").";
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error?.message || fallback);
    }
    if (isTensor) {
      tensor = await response.json();
      el("empty").hidden = true;
      el("tensor-result").hidden = false;
      el("tensor-title").textContent = tensor.output;
      el("tensor-shape").textContent = tensor.shape.join(" × ") + " · " + tensor.values.length + " values";
      const preview = tensor.values.slice(0, 32).map((value) => Number(value.toPrecision(6)));
      el("tensor-preview").textContent = JSON.stringify(preview, null, 2) + (tensor.values.length > 32 ? "\n…" : "");
      status("Tensor ready.");
    } else {
      let finished;
      await streamEvents(response.body, (item) => {
        if (item.error) throw new Error(item.error.message);
        if (item.event === "error" || item.reason === "error")
          throw new Error(item.message || "Inference failed.");
        if (item.event === "done") finished = item;
        if (item.text) reply += item.text;
        if (item.token_id !== undefined) tokens.push(item.token_id);
        const atBottom = results.scrollHeight - results.clientHeight - results.scrollTop < 70;
        outputMessage.body.textContent = model.text ? (reply || "Waiting for text…") : tokens.join(", ");
        if (atBottom) scrollToEnd();
      });
      if (!finished || finished.reason === "cancelled") throw new Error("Generation ended early.");
      outputMessage.body.textContent = model.text ? (reply || "No text generated.") : tokens.join(", ");
      recordReply();
      status(finished.emitted + " generated tokens" + (finished.reason === "length" ? " · Limit reached" : ""));
    }
  } catch (error) {
    if (!isTensor && (!controller.signal.aborted || (!reply && (model.text || !tokens.length)))) {
      inputMessage.group.remove();
      outputMessage.group.remove();
      el("empty").hidden = el("transcript").children.length > 0;
    } else if (!isTensor) {
      recordReply();
    }
    status(controller.signal.aborted ? "Stopped." : error.message, !controller.signal.aborted);
  } finally {
    active = null;
    busy(false);
    prompt.focus({ preventScroll: true });
  }
});

el("stop").addEventListener("click", () => active?.abort());
el("reset").addEventListener("click", () => {
  history = [];
  showMode("chat");
  prompt.focus();
});
el("input-kind").addEventListener("change", inputControls);
el("download").addEventListener("click", () => {
  if (!tensor) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(tensor)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = tensor.output + ".json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});
for (const button of modeButtons)
  button.addEventListener("click", () => showMode(button.dataset.mode));

async function start() {
  try {
    const response = await fetch("/model");
    if (!response.ok) throw new Error("Could not load model capabilities.");
    model = await response.json();
    const available = { chat: model.chat, completion: model.generation, tensor: model.outputs.length > 0 };
    for (const button of modeButtons) button.hidden = !available[button.dataset.mode];
    const preferred = Object.keys(available).find((item) => available[item]);
    if (!preferred) throw new Error("The checkpoint exposes no supported outputs.");
    const outputs = model.outputs;
    const selector = el("output");
    for (const output of outputs) selector.add(new Option(output.name, output.name));
    if (outputs.some((item) => item.name === "sentence_embedding")) selector.value = "sentence_embedding";
    el("max-tokens").max = Math.min(model.limits.max_output_tokens, model.generation_context || 1);
    el("max-tokens").value = Math.max(1, Math.min(128, Number(el("max-tokens").max)));
    el("model-id").textContent = model.id;
    el("architecture").textContent = model.architecture;
    el("context").textContent = (model.generation_context || model.context) + " tokens";
    el("modes").hidden = form.hidden = el("settings").hidden = false;
    el("connection").classList.add("connected");
    el("connection").lastElementChild.textContent = "Local model";
    showMode(preferred);
  } catch (error) {
    el("heading").textContent = "Model unavailable";
    el("subtitle").textContent = error.message;
    el("connection").lastElementChild.textContent = "Disconnected";
    status("Start the server with a supported checkpoint, then reload this page.", true);
  }
}
start();
