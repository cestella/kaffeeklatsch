package com.caseystella.llm.embeddings;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class HuggingFaceBiEncoder implements IBiEncoder {

  private HuggingFaceTokenizer tokenizer = null;
  private Path onnxLocation;


  public HuggingFaceBiEncoder(Path modelPath) throws IOException {
    this.tokenizer = HuggingFaceTokenizer.newInstance(modelPath);
    this.onnxLocation = HuggingFaceUtil.findModelFile(modelPath);
  }

  @Override
  public Iterable<float[]> embed(String[] sentences, Optional<OrtEnvironment> environmentOpt)
      throws OrtException {
    Encoding[] encodings = tokenizer.batchEncode(sentences);
    OrtEnvironment environment = environmentOpt.orElse(OrtEnvironment.getEnvironment());

    OrtSession session =
        environment.createSession(onnxLocation.toString(), new OrtSession.SessionOptions());
    Set<String> inputParams = session.getInputNames();
    Map<String, OnnxTensor> inputs = new HashMap<>();
    long[] shape = {encodings.length, encodings[0].getAttentionMask().length};
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
      int dimension = outputBuffer.capacity() / sentences.length;
      ArrayList<float[]> ret = new ArrayList<>(sentences.length);
      for (int i = 0; i < sentences.length; ++i) {
        float[] embedding = new float[dimension];
        outputBuffer.get(embedding);
        ret.add(embedding);
      }
      return ret;
    } finally {
      session.close();
      if (environmentOpt.isEmpty()) {
        environment.close();
      }
    }
  }
}
