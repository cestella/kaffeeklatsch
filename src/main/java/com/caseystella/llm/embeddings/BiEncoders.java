package com.caseystella.llm.embeddings;

import java.nio.file.Path;

public enum BiEncoders {
  ALL_MPNET_BASE_V2("sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2.onnx");
  String modelName;
  String onnxName;

  BiEncoders(String modelName, String onnxName) {
    this.modelName = modelName;
    this.onnxName = onnxName;
  }

  public IBiEncoder create(Path embeddingModelDir) {
    return new HuggingFaceBiEncoder(modelName, Path.of(embeddingModelDir.toString(), onnxName));
  }
}
