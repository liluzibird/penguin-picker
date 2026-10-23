/*
  Penguin Picker 2000 local model worker.
  CPU/WASM inference runs here so the questionnaire page stays responsive.
*/

var transformersModule = null;
var generator = null;
var loadingPromise = null;

function errorMessage(error) {
  return error && error.message ? error.message : String(error || "Unknown worker error");
}

function sendProgress(requestId, progress) {
  var percent = progress && typeof progress.progress === "number" ? progress.progress : 0;
  var detail = progress && (progress.text || progress.status || progress.file) ? (progress.text || progress.status || progress.file) : "";
  self.postMessage({
    type: "progress",
    requestId: requestId,
    progress: percent,
    detail: detail
  });
}

async function getGenerator(requestId, modelId, dtype) {
  if (generator) return generator;
  if (loadingPromise) {
    await loadingPromise;
    return generator;
  }

  loadingPromise = (async function () {
    self.postMessage({
      type: "status",
      requestId: requestId,
      message: "LOADING THE LOCAL CPU/WASM MODEL IN A BACKGROUND WORKER..."
    });
    transformersModule = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1");
    generator = await transformersModule.pipeline("text-generation", modelId, {
      dtype: dtype || "q8",
      progress_callback: function (progress) {
        sendProgress(requestId, progress);
      }
    });
  })();

  try {
    await loadingPromise;
    return generator;
  } finally {
    loadingPromise = null;
  }
}

self.addEventListener("message", async function (event) {
  var message = event.data || {};
  var requestId = message.requestId;

  try {
    var model = await getGenerator(requestId, message.modelId || "onnx-community/SmolLM2-135M-Instruct-ONNX-MHA", message.dtype || "q8");

    if (message.type === "load") {
      self.postMessage({ type: "ready", requestId: requestId });
      return;
    }

    if (message.type === "generate") {
      self.postMessage({
        type: "status",
        requestId: requestId,
        message: "GENERATING THE AI EXPLANATION IN THE BACKGROUND. The page remains responsive."
      });
      var requestedTokens = Number(message.maxNewTokens) || 192;
      var hasChatTemplate = Boolean(model.tokenizer && model.tokenizer.chat_template);
      var input = hasChatTemplate && Array.isArray(message.messages) ? message.messages : (message.prompt || "");
      var generationOptions = {
        max_new_tokens: Math.min(Math.max(requestedTokens, 96), 192),
        do_sample: false,
        return_full_text: false
      };
      var output;
      try {
        output = await model(input, generationOptions);
      } catch (generationError) {
        if (hasChatTemplate && Array.isArray(message.messages) && message.prompt) {
          self.postMessage({
            type: "status",
            requestId: requestId,
            message: "CHAT FORMAT FAILED; RETRYING THE SAME LOCAL MODEL WITH A PLAIN-TEXT PROMPT."
          });
          output = await model(message.prompt, generationOptions);
        } else {
          throw generationError;
        }
      }
      self.postMessage({ type: "result", requestId: requestId, output: output });
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      requestId: requestId,
      message: errorMessage(error)
    });
  }
});
