package com.caseystella.llm.embeddings;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.util.PairList;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class HuggingFaceCrossEncoder {
  private HuggingFaceTokenizer tokenizer = null;
  private Path onnxLocation;

  public HuggingFaceCrossEncoder(HuggingFaceTokenizer tokenizer, Path onnxLocation) {
    this.tokenizer = tokenizer;
    this.onnxLocation = onnxLocation;
  }

  public HuggingFaceCrossEncoder(String modelName, Path onnxLocation) {
    this(HuggingFaceUtil.resolveTokenizer(modelName, onnxLocation), onnxLocation);
  }

  public float[] embed(PairList<String, String> sentences, Optional<OrtEnvironment> environmentOpt)
      throws OrtException {
    Encoding[] encodings = tokenizer.batchEncode(sentences);
    OrtEnvironment environment = environmentOpt.orElse(OrtEnvironment.getEnvironment());

    OrtSession session =
        environment.createSession(onnxLocation.toString(), new OrtSession.SessionOptions());
    Set<String> inputParams = session.getInputNames();
    Map<String, OnnxTensor> inputs = new HashMap<>();
    long[] shape = {sentences.size(), encodings[0].getIds().length};
    if (inputParams.contains("input_ids")) {
      inputs.put(
          "input_ids", OnnxUtil.computeInput(encodings, shape, Encoding::getIds, environment));
    }
    if (inputParams.contains("attention_mask")) {
      inputs.put(
          "attention_mask",
          OnnxUtil.computeInput(encodings, shape, Encoding::getAttentionMask, environment));
    }
    if (inputParams.contains("token_type_ids")) {
      inputs.put(
          "token_type_ids",
          OnnxUtil.computeInput(encodings, shape, Encoding::getTypeIds, environment));
    }
    try (OrtSession.Result results = session.run(inputs)) {
      OnnxTensor outputTensor = (OnnxTensor) results.get(0);
      FloatBuffer outputBuffer = outputTensor.getFloatBuffer();

      float[] logits = new float[sentences.size()];
      for (int i = 0; i < sentences.size(); ++i) {
        float[] logit = new float[1];
        outputBuffer.get(logit);
        logits[i] = logit[0];
      }
      return logits;
    } finally {
      session.close();
      if (environmentOpt.isEmpty()) {
        environment.close();
      }
    }
  }
}
