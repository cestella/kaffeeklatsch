package com.caseystella.llm.embeddings;

import java.nio.file.Path;

public enum CrossEncoders {
  MS_MARCO_MiniLM_L6_v2(
      "cross-encoder/msmarco-MiniLM-L6-v2", "ce-msmarco-MiniLM-L6-v2/pytorch_model.onnx");
  String modelName;
  String onnxName;

  CrossEncoders(String modelName, String onnxName) {
    this.modelName = modelName;
    this.onnxName = onnxName;
  }

  public HuggingFaceCrossEncoder create(Path embeddingModelDir) {
    return new HuggingFaceCrossEncoder(modelName, Path.of(embeddingModelDir.toString(), onnxName));
  }
}
