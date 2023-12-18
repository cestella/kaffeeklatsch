package com.caseystella.llm.embeddings;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import java.io.IOException;
import java.nio.file.Path;

public class HuggingFaceUtil {

  public static HuggingFaceTokenizer resolveTokenizer(String modelName, Path onnxLocation) {
    try {
      return HuggingFaceTokenizer.newInstance(onnxLocation.toFile().getParentFile().toPath());
    } catch (IOException e) {
      return HuggingFaceTokenizer.newInstance(modelName);
    }
  }
}
